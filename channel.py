"""
channel.py — the network emulator (Layer 0, item 1).

This is the FIRST thing to build. It models the "network" that sits between
a sender and a receiver. A transport protocol with no impaired network has
nothing to react to, so this comes before the protocol and before any AI.

Model (kept deliberately simple for v1 — you'll deepen it later):

    sender --> [ propagation delay ] --> [ bottleneck: rate-limited finite queue ] --> receiver

Three sources of "interesting" behaviour, the things a congestion controller
must cope with:
  1. Propagation delay  -> packets take time to arrive (this creates RTT).
  2. Bottleneck rate     -> the link drains at a fixed bytes/sec. Send faster
                            than this and the queue grows.
  3. Finite buffer       -> when the queue is full, new packets are DROPPED
                            (this is congestion loss). Plus optional random loss.

It's a DISCRETE-EVENT model: we don't run in real wall-clock time, we advance
a virtual clock and process events in time order. This makes runs fast,
deterministic, and reproducible (critical for thesis-grade experiments).
"""

from __future__ import annotations
import heapq
from dataclasses import dataclass, field
import random


@dataclass(order=True)
class _Event:
    # Heap orders by time first; tie-break by an incrementing seq so equal-time
    # events keep insertion order and never try to compare the payloads.
    time: float
    seq: int
    payload: object = field(compare=False)


class Packet:
    """Minimal packet for the channel. The real protocol will subclass/extend
    this with seq/ack numbers later — for now we just need a size and an id."""
    __slots__ = ("pid", "size_bytes", "send_time")

    def __init__(self, pid: int, size_bytes: int, send_time: float):
        self.pid = pid
        self.size_bytes = size_bytes
        self.send_time = send_time

    def __repr__(self):
        return f"Packet(pid={self.pid}, size={self.size_bytes}B)"


class Channel:
    """A one-directional bottleneck link with delay, a finite queue, and loss.

    Parameters
    ----------
    bandwidth_bps : float
        Bottleneck drain rate in BITS per second. e.g. 10e6 = 10 Mbps.
    propagation_delay_s : float
        One-way propagation delay in seconds. e.g. 0.02 = 20 ms.
    buffer_bytes : int
        Max bytes that can sit in the bottleneck queue. Overflow => drop.
    loss_rate : float
        Independent random drop probability in [0, 1], on TOP of buffer drops.
    seed : int | None
        RNG seed for reproducible loss patterns.
    """

    def __init__(
        self,
        bandwidth_bps: float = 10e6,
        propagation_delay_s: float = 0.02,
        buffer_bytes: int = 64_000,
        loss_rate: float = 0.0,
        seed: int | None = 42,
    ):
        self.bandwidth_bps = bandwidth_bps
        self.propagation_delay_s = propagation_delay_s
        self.buffer_bytes = buffer_bytes
        self.loss_rate = loss_rate
        self._rng = random.Random(seed)

        # Event queue (min-heap by time) and a monotonically increasing counter.
        self._events: list[_Event] = []
        self._counter = 0
        self.now = 0.0

        # Bottleneck state.
        self._queue_bytes = 0          # bytes currently buffered
        self._link_busy_until = 0.0    # when the link finishes draining current backlog

        # Stats (your first telemetry — these become the structured log later).
        self.stats = {
            "sent": 0,
            "delivered": 0,
            "dropped_buffer": 0,
            "dropped_random": 0,
            "max_queue_bytes": 0,
        }

        # The receiver callback: set this to get delivered packets.
        self.on_deliver = None  # callable(packet, arrival_time)

    # --- event plumbing -------------------------------------------------

    def _schedule(self, time: float, payload):
        heapq.heappush(self._events, _Event(time, self._counter, payload))
        self._counter += 1

    def _serialization_time(self, size_bytes: int) -> float:
        """Time to push `size_bytes` onto the link at the bottleneck rate."""
        return (size_bytes * 8) / self.bandwidth_bps

    # --- public API -----------------------------------------------------

    def send(self, packet: Packet):
        """Sender hands a packet to the channel at time self.now."""
        self.stats["sent"] += 1

        # 1. Independent random loss (models flaky wireless, etc.)
        if self.loss_rate > 0 and self._rng.random() < self.loss_rate:
            self.stats["dropped_random"] += 1
            return

        # 2. Buffer overflow check (congestion loss — the important one).
        if self._queue_bytes + packet.size_bytes > self.buffer_bytes:
            self.stats["dropped_buffer"] += 1
            return

        # 3. Admit to queue. Compute when the link can start draining it:
        #    either now (link idle) or after the current backlog clears.
        self._queue_bytes += packet.size_bytes
        self.stats["max_queue_bytes"] = max(
            self.stats["max_queue_bytes"], self._queue_bytes
        )

        start_drain = max(self.now, self._link_busy_until)
        serialize = self._serialization_time(packet.size_bytes)
        depart_time = start_drain + serialize
        self._link_busy_until = depart_time

        # Arrival = departure from bottleneck + propagation delay.
        arrival_time = depart_time + self.propagation_delay_s
        self._schedule(arrival_time, ("deliver", packet))

        # Free the queue space at departure (packet has left the buffer).
        self._schedule(depart_time, ("dequeue", packet.size_bytes))

    def run_until(self, end_time: float):
        """Advance the virtual clock, processing events up to end_time."""
        while self._events and self._events[0].time <= end_time:
            ev = heapq.heappop(self._events)
            self.now = ev.time
            kind = ev.payload[0]

            if kind == "dequeue":
                self._queue_bytes -= ev.payload[1]
            elif kind == "deliver":
                packet = ev.payload[1]
                self.stats["delivered"] += 1
                if self.on_deliver:
                    self.on_deliver(packet, self.now)

        self.now = end_time

    @property
    def queue_bytes(self) -> int:
        return self._queue_bytes