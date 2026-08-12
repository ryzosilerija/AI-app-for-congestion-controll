"""
demo_controllers.py — window-based congestion controllers for the live demo.

Each controller manages a CWND (congestion window in bytes). The server only
sends when bytes_inflight < cwnd. This gives the controller real leverage:
  * shrink cwnd -> server stops sending -> queue drains -> RTT drops
  * grow cwnd -> more packets in flight -> higher throughput (but risk bufferbloat)

This is the same mechanism as real TCP/CUBIC/your trained agent.
"""


class Blind:
    """No control — infinite window, send as fast as possible."""
    name = "BLIND (no control)"
    def __init__(self):
        self.cwnd = 10_000_000     # effectively infinite
    def update(self, srtt, rtt_min, loss_events, bytes_acked, **kw):
        pass


class CubicLike:
    """Loss-based AIMD: grows the window until loss, then cuts it.
    Ignores latency — will fill the buffer (bufferbloat)."""
    name = "CUBIC-like (loss-based)"
    def __init__(self):
        self.cwnd = 20_000         # start moderate
    def update(self, srtt, rtt_min, loss_events, bytes_acked, **kw):
        if loss_events > 0:
            self.cwnd = max(2000, int(self.cwnd * 0.5))    # multiplicative decrease
        else:
            self.cwnd += max(500, bytes_acked // 4)        # additive increase (aggressive)


class DelayCC:
    """Delay-based: backs off when RTT climbs above minimum (keeps latency low)."""
    name = "DELAY-CC (latency-aware)"
    def __init__(self):
        self.cwnd = 20_000
    def update(self, srtt, rtt_min, loss_events, bytes_acked, **kw):
        if loss_events > 0:
            self.cwnd = max(2000, int(self.cwnd * 0.5))
            return
        if srtt <= 0 or rtt_min <= 0:
            return
        inflation = srtt / rtt_min
        if inflation > 1.2:         # queue building -> shrink to drain it
            self.cwnd = max(5000, int(self.cwnd * 0.9))
        elif inflation < 1.05:      # queue empty -> grow cautiously
            self.cwnd += max(500, bytes_acked // 4)


def make(name):
    return {"blind": Blind, "cubic": CubicLike, "delay": DelayCC}[name]()