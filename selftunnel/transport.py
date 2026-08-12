"""A small reliable UDP transport with a congestion-control seam.

This is deliberately a *reference* transport so the tunnel runs standalone.
It gives you: sequenced packets, cumulative acks, RTT sampling, retransmit on
timeout, and sending gated by a CongestionController. Your PyQUIC-RL transport
can replace this wholesale — the tunnel only needs the Transport API at the
bottom: connect/accept, send_stream(data), recv_stream() and close.

Wire format (per UDP datagram): one header byte + fields.
  DATA:  b'D' | seq(4) | length(2) | payload
  ACK:   b'A' | ack_seq(4)          # cumulative: "I have everything <= ack_seq"

Not production-grade (no encryption, simple loss detection) — it's the seam,
not the finished protocol.
"""
import socket
import struct
import threading
import time
from collections import deque

from congestion import CongestionController, AIMD

MSS = 1200
DATA, ACK = b'D', b'A'


class Transport:
    def __init__(self, sock: socket.socket, peer, cc: CongestionController | None = None):
        self.sock = sock
        self.peer = peer
        self.cc = cc or AIMD()

        self.send_seq = 0
        self.recv_seq = 0                       # highest in-order seq received
        self.unacked = {}                       # seq -> (payload, send_time, retries)
        self.inflight = 0
        self.send_buf = deque()                 # bytes waiting for the cwnd
        self.recv_buf = deque()                 # reassembled payloads for the app

        self.srtt = 0.1
        self.rttvar = 0.05
        self.rto = 0.2

        self._lock = threading.Lock()
        self._closed = False
        self._rx = threading.Thread(target=self._rx_loop, daemon=True)
        self._tx = threading.Thread(target=self._tx_loop, daemon=True)
        self._rx.start()
        self._tx.start()

    # --- public API the tunnel uses ---

    def send_stream(self, data: bytes):
        with self._lock:
            for i in range(0, len(data), MSS):
                self.send_buf.append(data[i:i + MSS])

    def recv_stream(self, timeout=0.5) -> bytes | None:
        deadline = time.time() + timeout
        while time.time() < deadline:
            with self._lock:
                if self.recv_buf:
                    return self.recv_buf.popleft()
            time.sleep(0.001)
        return None

    def close(self):
        self._closed = True

    # --- internals ---

    def _rx_loop(self):
        while not self._closed:
            try:
                self.sock.settimeout(0.2)
                pkt, addr = self.sock.recvfrom(2048)
            except socket.timeout:
                continue
            except OSError:
                break
            if not pkt:
                continue
            kind = pkt[:1]
            if kind == DATA:
                seq, length = struct.unpack('!IH', pkt[1:7])
                payload = pkt[7:7 + length]
                with self._lock:
                    if seq == self.recv_seq:            # in order
                        self.recv_seq += 1
                        self.recv_buf.append(payload)
                    # cumulative ack of what we have
                    self._send_ack(self.recv_seq - 1)
            elif kind == ACK:
                (ack_seq,) = struct.unpack('!I', pkt[1:5])
                self._handle_ack(ack_seq)

    def _send_ack(self, ack_seq):
        if ack_seq < 0:
            return
        self.sock.sendto(ACK + struct.pack('!I', ack_seq), self.peer)

    def _handle_ack(self, ack_seq):
        now = time.time()
        with self._lock:
            for seq in list(self.unacked):
                if seq <= ack_seq:
                    payload, sent, _ = self.unacked.pop(seq)
                    self.inflight -= len(payload)
                    rtt = now - sent
                    self._update_rtt(rtt)
                    self.cc.on_ack(rtt, len(payload))

    def _update_rtt(self, sample):
        # Jacobson/Karels
        self.rttvar = 0.75 * self.rttvar + 0.25 * abs(self.srtt - sample)
        self.srtt = 0.875 * self.srtt + 0.125 * sample
        self.rto = max(0.05, self.srtt + 4 * self.rttvar)

    def _tx_loop(self):
        while not self._closed:
            now = time.time()
            with self._lock:
                # retransmit timed-out packets, signal loss to CC
                for seq, (payload, sent, retries) in list(self.unacked.items()):
                    if now - sent > self.rto:
                        self.cc.on_loss()
                        self._raw_send(seq, payload)
                        self.unacked[seq] = (payload, now, retries + 1)
                # send new data while the window allows
                while self.send_buf and self.cc.can_send(self.inflight):
                    payload = self.send_buf.popleft()
                    seq = self.send_seq
                    self.send_seq += 1
                    self.unacked[seq] = (payload, now, 0)
                    self.inflight += len(payload)
                    self._raw_send(seq, payload)
            time.sleep(0.0005)

    def _raw_send(self, seq, payload):
        hdr = DATA + struct.pack('!IH', seq, len(payload))
        self.sock.sendto(hdr + payload, self.peer)
