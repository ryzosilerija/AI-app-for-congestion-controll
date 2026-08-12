"""
udp_link.py — two-way UDP link with seq numbers, ACKs, RTT/loss measurement,
bytes-in-flight tracking for window-based flow control, and a command channel.

The key addition: bytes_inflight tracks how many bytes are sent but not yet ACKed.
The server uses this to implement window-based pacing: only send when
bytes_inflight < cwnd. This gives controllers real leverage over latency —
shrinking the window stops new sends, lets the queue drain, and RTT drops.
"""
from __future__ import annotations
import socket
import struct
import time

TYPE_DATA = 0
TYPE_ACK = 1
TYPE_CMD = 2

DATA_HDR = "!BId"      # type(1) + seq(4) + timestamp(8) = 13 bytes
ACK_FMT = "!BId"
CMD_FMT = "!BBh"


class UDPLink:
    def __init__(self, sock, peer_addr=None):
        self.sock = sock
        self.sock.setblocking(False)
        self.peer = peer_addr
        self.sock_sendto_override = None
        self.seq = 0
        self.sent = {}              # seq -> (send_ts, pkt_len)
        self.acked = 0
        self.lost = 0
        self.srtt = None
        self.rtt_min = float("inf")
        self.bytes_inflight = 0     # sent but not yet ACKed
        self._loss_events = 0       # count of loss detections

    # ---- sender ----
    def send_data(self, payload: bytes):
        ts = time.monotonic()
        pkt = struct.pack(DATA_HDR, TYPE_DATA, self.seq, ts) + payload
        if self.sock_sendto_override is not None:
            self.sock_sendto_override(pkt, self.peer)
        else:
            self.sock.sendto(pkt, self.peer)
        self.sent[self.seq] = (ts, len(pkt))
        self.bytes_inflight += len(pkt)
        self.seq += 1
        # timeout old unacked packets (> 3 seconds = lost)
        now = ts
        lost_seqs = [s for s, (t, l) in self.sent.items()
                     if now - t > 3.0 and s != self.seq - 1]
        for s in lost_seqs:
            _, plen = self.sent.pop(s)
            self.bytes_inflight = max(0, self.bytes_inflight - plen)
            self.lost += 1
            self._loss_events += 1
        return self.seq - 1

    def _on_ack(self, seq, now):
        if seq in self.sent:
            send_ts, plen = self.sent.pop(seq)
            rtt = now - send_ts
            self.acked += 1
            self.bytes_inflight = max(0, self.bytes_inflight - plen)
            self.rtt_min = min(self.rtt_min, rtt)
            self.srtt = rtt if self.srtt is None else 0.875 * self.srtt + 0.125 * rtt

    def pop_loss_events(self):
        """Return and reset the loss event count since last call."""
        n = self._loss_events
        self._loss_events = 0
        return n

    # ---- command channel ----
    def send_cmd(self, cmd: int, arg: int = 0):
        self.sock.sendto(struct.pack(CMD_FMT, TYPE_CMD, cmd, arg), self.peer)

    # ---- receiver ----
    def _send_ack(self, seq, echo_ts, addr):
        self.sock.sendto(struct.pack(ACK_FMT, TYPE_ACK, seq, echo_ts), addr)

    def poll(self, on_data=None, on_cmd=None):
        now = time.monotonic()
        while True:
            try:
                pkt, addr = self.sock.recvfrom(65535)
            except BlockingIOError:
                break
            except (ConnectionResetError, OSError):
                break
            if not pkt:
                continue
            ptype = pkt[0]
            if ptype == TYPE_DATA:
                _, seq, send_ts = struct.unpack_from(DATA_HDR, pkt, 0)
                payload = pkt[struct.calcsize(DATA_HDR):]
                self._send_ack(seq, send_ts, addr)
                if self.peer is None:
                    self.peer = addr
                if on_data is not None:
                    on_data(payload)
            elif ptype == TYPE_ACK:
                _, seq, echo_ts = struct.unpack_from(ACK_FMT, pkt, 0)
                self._on_ack(seq, now)
            elif ptype == TYPE_CMD:
                _, cmd, arg = struct.unpack_from(CMD_FMT, pkt, 0)
                if self.peer is None:
                    self.peer = addr
                if on_cmd is not None:
                    on_cmd(cmd, arg)

    def stats(self):
        srtt_ms = (self.srtt * 1000) if self.srtt else 0.0
        total = self.acked + self.lost
        return dict(acked=self.acked, srtt_ms=srtt_ms,
                    rtt_min_ms=self.rtt_min * 1000 if self.rtt_min != float("inf") else 0.0,
                    loss_pct=100 * self.lost / max(1, total),
                    bytes_inflight=self.bytes_inflight)