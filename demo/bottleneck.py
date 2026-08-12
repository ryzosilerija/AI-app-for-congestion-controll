"""
bottleneck.py — a software network bottleneck (a simulated pipe) so congestion
can actually happen, on localhost or any link that's too fast to congest on its own.

Models a fixed-capacity link with a finite queue:
  * packets enter a queue
  * the queue drains at a fixed RATE (bytes/sec) -> the bandwidth limit
  * packets wait in the queue -> queuing DELAY (this is what bloats latency)
  * if the queue is full, packets are DROPPED -> loss
  * an added base propagation DELAY simulates distance

This is the "corner" that makes a good controller look different from a bad one:
send faster than `rate` and the queue fills -> latency climbs and loss appears,
exactly the situation your controller was trained to manage.

Usage: the sender pushes outgoing packets through a Bottleneck instead of
straight to the socket; the bottleneck releases them to the socket on schedule.
"""
from __future__ import annotations
import time
import collections


class Bottleneck:
    def __init__(self, rate_mbps=2.0, base_delay_ms=20.0, queue_bytes=30_000):
        self.rate_bps = rate_mbps * 1e6 / 8      # bytes/sec
        self.base_delay = base_delay_ms / 1000.0
        self.queue_max = queue_bytes
        self.queue = collections.deque()          # (release_time, dest_addr, data)
        self.queue_bytes = 0
        self.next_free = time.monotonic()         # when the link can send next
        self.dropped = 0
        self.passed = 0

    def set_rate(self, rate_mbps):
        self.rate_bps = max(0.05e6, rate_mbps * 1e6) / 8

    def offer(self, data, dest, now=None):
        """Try to enqueue an outgoing packet. Returns True if queued, False if
        dropped (queue full = congestion loss)."""
        if now is None:
            now = time.monotonic()
        if self.queue_bytes + len(data) > self.queue_max:
            self.dropped += 1
            return False
        # schedule release: after the link is free, plus this packet's serialization
        serialize = len(data) / self.rate_bps
        start = max(now, self.next_free)
        release = start + serialize + self.base_delay
        self.next_free = start + serialize
        self.queue.append((release, dest, data))
        self.queue_bytes += len(data)
        return True

    def drain(self, send_fn, now=None):
        """Release any packets whose scheduled time has arrived, via send_fn(data,dest)."""
        if now is None:
            now = time.monotonic()
        while self.queue and self.queue[0][0] <= now:
            release, dest, data = self.queue.popleft()
            self.queue_bytes -= len(data)
            send_fn(data, dest)
            self.passed += 1

    def stats(self):
        total = self.passed + self.dropped
        return dict(queue_bytes=self.queue_bytes,
                    drop_pct=100 * self.dropped / max(1, total),
                    dropped=self.dropped, passed=self.passed)