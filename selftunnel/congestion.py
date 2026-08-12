"""Congestion-control seam.

The tunnel sends data over a UDP transport whose pacing is decided by a
CongestionController. This module defines the interface your PyQUIC-RL agent
plugs into, plus a working AIMD default so the tunnel runs end to end today.

To use your own controller, implement this interface and pass an instance to
the transport (see transport.py). The methods mirror what a congestion-control
plugin needs: react to acks and losses, and gate/pace sending.
"""
import time


class CongestionController:
    """Interface. Subclass this (or your RL agent) and implement the hooks."""

    def can_send(self, inflight_bytes: int) -> bool:
        """May we send another packet right now, given bytes in flight?"""
        raise NotImplementedError

    def on_ack(self, rtt_sample: float, acked_bytes: int) -> None:
        """An ack arrived. rtt_sample in seconds."""
        raise NotImplementedError

    def on_loss(self) -> None:
        """A loss was detected (timeout or duplicate-ack style signal)."""
        raise NotImplementedError

    @property
    def window_bytes(self) -> int:
        """Current congestion window in bytes."""
        raise NotImplementedError


class AIMD(CongestionController):
    """A minimal, working AIMD controller — slow start then additive increase,
    multiplicative decrease on loss. Good enough to run the tunnel; your RL
    agent replaces it by implementing the same interface."""

    def __init__(self, mss: int = 1200, init_cwnd_packets: int = 10):
        self.mss = mss
        self.cwnd = init_cwnd_packets * mss
        self.ssthresh = 64 * mss
        self.in_slow_start = True
        self._acked_in_rtt = 0

    @property
    def window_bytes(self) -> int:
        return int(self.cwnd)

    def can_send(self, inflight_bytes: int) -> bool:
        return inflight_bytes < self.cwnd

    def on_ack(self, rtt_sample: float, acked_bytes: int) -> None:
        if self.in_slow_start:
            self.cwnd += self.mss            # exponential-ish growth
            if self.cwnd >= self.ssthresh:
                self.in_slow_start = False
        else:
            # additive increase: +MSS per RTT, approximated per-ack
            self._acked_in_rtt += acked_bytes
            if self._acked_in_rtt >= self.cwnd:
                self.cwnd += self.mss
                self._acked_in_rtt = 0

    def on_loss(self) -> None:
        self.ssthresh = max(2 * self.mss, self.cwnd // 2)
        self.cwnd = self.ssthresh
        self.in_slow_start = False


# Registry so the CLI can select a controller by name.
CONTROLLERS = {
    "aimd": AIMD,
    # "rl": YourRLController,   # <- register your PyQUIC-RL agent here
}


def make_controller(name: str) -> CongestionController:
    if name not in CONTROLLERS:
        raise SystemExit(f"unknown congestion control '{name}'. "
                         f"available: {', '.join(CONTROLLERS)}")
    return CONTROLLERS[name]()
