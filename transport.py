"""
transport.py — the reliable transport over the Channel (items 4-7).

Turns blind packet-blasting into a real protocol:
  * Packets carry seq numbers + timestamps.
  * Receiver sends ACKs back through a SECOND channel (the return path).
  * Sender measures RTT from acked packets (Jacobson/Karels EWMA -> SRTT, RTO).
  * Loss is detected two ways: retransmission TIMEOUT, and 3 duplicate ACKs
    (fast retransmit).
  * Sender only keeps `cwnd` bytes in flight (cwnd comes from the controller).

The controller (AIMD now, AI later) is handed in and driven via on_ack/on_loss.
This is a SENDER-side congestion-control model, which is where ~all the
interesting CC logic lives.
"""

from __future__ import annotations
from dataclasses import dataclass
from channel import Channel, Packet


@dataclass
class DataPacket(Packet):
    """A data packet with a sequence number (in bytes) and a send timestamp."""
    def __init__(self, seq: int, size_bytes: int, send_time: float,
                 is_retransmit: bool = False):
        super().__init__(pid=seq, size_bytes=size_bytes, send_time=send_time)
        self.seq = seq
        self.is_retransmit = is_retransmit


class AckPacket(Packet):
    """An ACK. Carries the cumulative ack number and echoes the data packet's
    send time so the sender can compute RTT on receipt."""
    def __init__(self, ack_seq: int, echo_send_time: float, now: float):
        super().__init__(pid=-ack_seq - 1, size_bytes=40, send_time=now)
        self.ack_seq = ack_seq
        self.echo_send_time = echo_send_time


class RttEstimator:
    """Jacobson/Karels RTT smoothing -> RTO. The same math real TCP uses."""
    def __init__(self, alpha=1/8, beta=1/4, min_rto=0.2, max_rto=10.0):
        self.alpha, self.beta = alpha, beta
        self.srtt = None
        self.rttvar = None
        self.min_rto, self.max_rto = min_rto, max_rto
        self.rtt_min = float("inf")     # tracked for the AI state vector later

    def update(self, sample: float):
        self.rtt_min = min(self.rtt_min, sample)
        if self.srtt is None:
            self.srtt = sample
            self.rttvar = sample / 2
        else:
            self.rttvar = (1 - self.beta) * self.rttvar + self.beta * abs(self.srtt - sample)
            self.srtt = (1 - self.alpha) * self.srtt + self.alpha * sample

    @property
    def rto(self) -> float:
        if self.srtt is None:
            return 1.0
        return min(max(self.srtt + 4 * self.rttvar, self.min_rto), self.max_rto)


class Sender:
    def __init__(self, controller, forward: Channel, reverse: Channel,
                 mss: int = 1200, total_bytes: int = 5_000_000):
        self.cc = controller
        self.forward = forward
        self.reverse = reverse
        self.mss = mss
        self.total_bytes = total_bytes

        self.next_seq = 0           # next byte offset to send
        self.send_base = 0          # oldest unacked byte (cumulative ack point)
        self.inflight = {}          # seq -> (send_time, size, deadline)
        self._inflight_bytes = 0    # incrementally maintained sum of inflight sizes
        self.rtt = RttEstimator()

        self._dup_ack_count = 0
        self._last_ack = -1
        self.acked_bytes = 0

        # telemetry samples: (time, cwnd_pkts, inflight_pkts, srtt, goodput_bps)
        self.trace = []

    @property
    def bytes_inflight(self) -> int:
        # Incrementally maintained (see _inflight_bytes) — O(1), not O(n).
        # A prior version summed the whole inflight dict on every _can_send()
        # call, which stalled for minutes/hours when a timid policy let the dict
        # grow large. Never re-sum here.
        return self._inflight_bytes

    def _can_send(self) -> bool:
        return (self._inflight_bytes + self.mss <= self.cc.cwnd
                and self.next_seq < self.total_bytes)

    def _transmit(self, seq: int, now: float, retransmit=False):
        size = min(self.mss, self.total_bytes - seq)
        if size <= 0:
            return
        # If this seq is already tracked (a retransmit), don't double-count bytes.
        if seq not in self.inflight:
            self._inflight_bytes += size
        pkt = DataPacket(seq, size, now, is_retransmit=retransmit)
        self.inflight[seq] = (now, size, now + self.rtt.rto)
        self.forward.send(pkt)

    def pump(self, now: float):
        """Send as many new packets as the window allows."""
        while self._can_send():
            self._transmit(self.next_seq, now)
            self.next_seq += min(self.mss, self.total_bytes - self.next_seq)

    def on_ack_received(self, ack: AckPacket, now: float):
        # RTT sample from the echoed timestamp.
        rtt_sample = now - ack.echo_send_time
        if rtt_sample > 0:
            self.rtt.update(rtt_sample)

        if ack.ack_seq > self.send_base:
            # New data acknowledged (cumulative). Remove acked segments.
            newly = 0
            for seq in sorted(list(self.inflight.keys())):
                if seq < ack.ack_seq:
                    _, sz, _ = self.inflight.pop(seq)
                    newly += sz
                    self._inflight_bytes -= sz      # keep counter in sync
            self.send_base = ack.ack_seq
            self.acked_bytes += newly
            self._dup_ack_count = 0
            self.cc.on_ack(newly, rtt_sample, now)
        elif ack.ack_seq == self.send_base:
            # Duplicate ACK -> possible loss. 3 dups = fast retransmit.
            self._dup_ack_count += 1
            if self._dup_ack_count == 3:
                self.cc.on_loss(now)
                self._transmit(self.send_base, now, retransmit=True)

    def check_timeouts(self, now: float):
        """Retransmission timeout: oldest unacked packet past its deadline."""
        for seq, (st, sz, deadline) in list(self.inflight.items()):
            if now >= deadline:
                self.cc.on_loss(now)
                self._transmit(seq, now, retransmit=True)
                # back off this entry's deadline to avoid a storm
                self.inflight[seq] = (st, sz, now + self.rtt.rto)
                break

    def record(self, now: float):
        gp = (self.acked_bytes * 8) / now if now > 0 else 0
        self.trace.append((
            now, self.cc.cwnd_packets, self.bytes_inflight / self.mss,
            self.rtt.srtt or 0.0, gp,
        ))

    @property
    def finished(self) -> bool:
        return self.send_base >= self.total_bytes


class Receiver:
    """Cumulative-ACK receiver. Acks the highest in-order byte received."""
    def __init__(self, reverse: Channel, mss: int = 1200):
        self.reverse = reverse
        self.mss = mss
        self.expected = 0
        self.received = set()

    def on_data_received(self, pkt: DataPacket, now: float):
        if pkt.seq == self.expected:
            self.expected += pkt.size_bytes
            # absorb any buffered contiguous segments
            while self.expected in self.received:
                self.received.discard(self.expected)
                self.expected += self.mss
        elif pkt.seq > self.expected:
            self.received.add(pkt.seq)
        ack = AckPacket(self.expected, pkt.send_time, now)
        self.reverse.send(ack)