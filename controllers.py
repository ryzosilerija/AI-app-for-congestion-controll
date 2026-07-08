"""
controllers.py — the pluggable congestion-control interface (item 8) + baselines.

THE KEY DESIGN DECISION of the whole project lives here. Every congestion
controller — the classic AIMD baseline, CUBIC later, and your AI agent later —
implements this ONE interface:

    on_ack(packet_acked, rtt_sample, now)   # called when an ACK arrives
    on_loss(now)                            # called when loss is detected
    cwnd                                    # property: window in BYTES

The transport calls these hooks and reads `cwnd` to decide how many bytes it's
allowed to have in flight. The controller never touches sockets, packets, or
timers directly — it only sees "an ack happened / a loss happened" and decides
the window. That clean separation is what lets you drop in an RL agent later
WITHOUT rewriting the protocol.
"""

from __future__ import annotations


class CongestionController:
    """Base class. Subclass this for every algorithm."""

    def __init__(self, mss: int = 1200, init_cwnd_packets: float = 2.0):
        self.mss = mss                       # max segment size (bytes/packet)
        self._cwnd = init_cwnd_packets * mss # window in BYTES

    @property
    def cwnd(self) -> float:
        return self._cwnd

    @property
    def cwnd_packets(self) -> float:
        return self._cwnd / self.mss

    def on_ack(self, bytes_acked: int, rtt_sample: float, now: float):
        raise NotImplementedError

    def on_loss(self, now: float):
        raise NotImplementedError


class FixedWindow(CongestionController):
    """Trivial baseline: never changes the window. Useful as a sanity control —
    shows what a NON-adaptive sender does (over- or under-shoots the link)."""

    def on_ack(self, bytes_acked, rtt_sample, now):
        pass

    def on_loss(self, now):
        pass


class AIMD(CongestionController):
    """Additive-Increase / Multiplicative-Decrease — the classic TCP Reno core.

    The famous 'sawtooth':
      * Every ACK in congestion avoidance grows cwnd by mss * (mss / cwnd),
        which sums to roughly +1 packet per RTT (additive increase).
      * On loss, cwnd is halved (multiplicative decrease) and we set the
        slow-start threshold there.
      * Below ssthresh we're in SLOW START: cwnd grows by 1 packet per ACK
        (exponential per RTT) to quickly find the link's capacity.

    This is your first REAL controller and the baseline your AI must beat.
    """

    def __init__(self, mss: int = 1200, init_cwnd_packets: float = 2.0,
                 beta: float = 0.5):
        super().__init__(mss, init_cwnd_packets)
        self.ssthresh = 64 * mss     # start high; first loss will set it properly
        self.beta = beta             # multiplicative decrease factor

    def on_ack(self, bytes_acked, rtt_sample, now):
        if self._cwnd < self.ssthresh:
            # Slow start: +1 MSS per ACK (exponential growth per RTT).
            self._cwnd += self.mss
        else:
            # Congestion avoidance: ~ +1 MSS per RTT (additive).
            self._cwnd += self.mss * (self.mss / self._cwnd)

    def on_loss(self, now):
        # Multiplicative decrease.
        self.ssthresh = max(self._cwnd * self.beta, 2 * self.mss)
        self._cwnd = self.ssthresh


class Cubic(CongestionController):
    """TCP CUBIC — Linux's default since 2006, the REAL modern baseline.

    Unlike AIMD's linear growth, CUBIC grows the window along a CUBIC function
    of time since the last loss:

        W(t) = C * (t - K)^3 + W_max

    where:
      W_max = window size just before the last loss (the target to probe back to)
      K     = cube-root( W_max * (1 - beta) / C )   (time to return to W_max)
      C     = scaling constant (aggressiveness), conventionally 0.4
      beta  = multiplicative decrease factor, conventionally 0.7 (gentler than
              AIMD's 0.5 — CUBIC gives up less window on loss)

    The shape: after a loss, growth is fast (concave) approaching W_max, slows
    near W_max (plateau, "probing" the same capacity), then accelerates again
    (convex) past W_max to discover new capacity. This is what makes CUBIC fill
    large pipes fast — and also what keeps the bottleneck buffer full, inflating
    latency. That latency cost is your AI's opening.

    Also implements TCP-friendly (Reno-equivalent) growth as a floor so it never
    underperforms Reno on small-RTT links.
    """

    def __init__(self, mss: int = 1200, init_cwnd_packets: float = 2.0,
                 C: float = 0.4, beta: float = 0.7):
        super().__init__(mss, init_cwnd_packets)
        self.C = C
        self.beta = beta
        self.ssthresh = 64 * mss
        self.W_max = 0.0          # window before last loss (bytes)
        self.K = 0.0              # time to reach W_max again
        self.epoch_start = None   # time of last loss (start of cubic epoch)
        self.W_tcp = 0.0          # the Reno-friendly estimate (bytes)

    def on_ack(self, bytes_acked, rtt_sample, now):
        if self._cwnd < self.ssthresh:
            # Slow start, same as Reno.
            self._cwnd += self.mss
            return

        # --- Congestion avoidance: the CUBIC update ---
        if self.epoch_start is None:
            # First ACK of a new epoch (just after a loss).
            self.epoch_start = now
            if self.W_max < self._cwnd:
                # rare: window grew past previous max without loss
                self.K = 0.0
                self.W_max = self._cwnd
            else:
                # K = cube_root( W_max * (1 - beta) / C ), in packet units
                wmax_pkts = self.W_max / self.mss
                self.K = (wmax_pkts * (1 - self.beta) / self.C) ** (1 / 3)
            self.W_tcp = self._cwnd

        t = now - self.epoch_start
        wmax_pkts = self.W_max / self.mss
        # CUBIC target window (in packets) at time t into the epoch.
        target_pkts = self.C * (t - self.K) ** 3 + wmax_pkts
        target = target_pkts * self.mss

        # TCP-friendly region: track what Reno would do, take the max.
        # Reno per-ACK increment ~ mss * mss / cwnd.
        self.W_tcp += self.mss * (self.mss / self._cwnd) * (bytes_acked / self.mss)

        cubic_cwnd = max(target, self._cwnd + (target - self._cwnd) * 0.5)
        self._cwnd = max(cubic_cwnd, self.W_tcp)

    def on_loss(self, now):
        self.epoch_start = None          # start a fresh cubic epoch next ACK
        self.W_max = self._cwnd          # remember the window we lost at
        self.ssthresh = max(self._cwnd * self.beta, 2 * self.mss)
        self._cwnd = self.ssthresh


class BBR(CongestionController):
    """BBR (Bottleneck Bandwidth and RTT) — Google's model-based controller.

    THE key difference from CUBIC/AIMD: BBR does NOT use loss as the congestion
    signal. Instead it MEASURES the pipe directly and paces to its 'sweet spot':

        BDP = estimated_bottleneck_bandwidth * min_RTT

    Sitting at the BDP means the pipe is full (high throughput) but the buffer is
    empty (low latency) — exactly the throughput/latency point a good learned
    controller also targets. That's why BBR is the real competition: it's the
    hand-engineered version of what your agent learns.

    Simplified phase machine (faithful to BBR's logic, adapted to this sim):
      STARTUP   : ramp fast (gain 2.89) to discover bandwidth; exit when
                  bandwidth stops growing (pipe found).
      DRAIN     : gain < 1 to drain the queue STARTUP overshot into.
      PROBE_BW  : steady state. Cycle pacing gains [1.25, 0.75, 1,1,1,1,1,1] to
                  periodically probe for more bandwidth then give it back.
      PROBE_RTT : every ~10s, cut cwnd briefly to re-measure the true min_RTT.

    Bandwidth is estimated as a windowed max of delivery-rate samples; min_RTT as
    a windowed min of RTT samples. cwnd is set to ~2*BDP (BBR's cwnd_gain=2).
    """

    STARTUP_GAIN = 2.89        # ~2/ln2, BBR's startup pacing/cwnd gain
    DRAIN_GAIN = 1.0 / 2.89
    CWND_GAIN = 2.0
    PROBE_BW_GAINS = [1.25, 0.75, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    RTPROP_WIN = 10.0          # min-RTT measurement window (s)
    PROBE_RTT_DUR = 0.2        # how long to hold reduced cwnd (s)

    def __init__(self, mss: int = 1200, init_cwnd_packets: float = 4.0):
        super().__init__(mss, init_cwnd_packets)
        self.mode = "STARTUP"
        self.btl_bw = 0.0               # bytes/sec, windowed max
        self.rt_prop = float("inf")     # seconds, windowed min RTT
        self._rt_prop_stamp = 0.0
        self._bw_samples = []           # recent (time, bw) for windowed max
        self._full_bw = 0.0             # for STARTUP exit detection
        self._full_bw_count = 0
        self._cycle_idx = 0
        self._cycle_stamp = 0.0
        self._probe_rtt_done = None
        self._acked_total = 0
        self._last_ack_time = 0.0

    def _update_bw(self, bytes_acked, now):
        # delivery-rate sample over the time since last ack batch
        dt = now - self._last_ack_time
        if dt > 0:
            rate = bytes_acked / dt
            self._bw_samples.append((now, rate))
        self._last_ack_time = now
        # keep ~ up to 10 RTTs of samples; windowed max
        horizon = now - max(self.rt_prop * 10, 0.5)
        self._bw_samples = [(t, r) for (t, r) in self._bw_samples if t >= horizon]
        if self._bw_samples:
            self.btl_bw = max(r for (_, r) in self._bw_samples)

    def _bdp(self):
        if self.btl_bw <= 0 or self.rt_prop == float("inf"):
            return 4 * self.mss
        return self.btl_bw * self.rt_prop

    def on_ack(self, bytes_acked, rtt_sample, now):
        # --- update min-RTT (rt_prop) over a sliding window ---
        if rtt_sample > 0 and (rtt_sample < self.rt_prop
                               or now - self._rt_prop_stamp > self.RTPROP_WIN):
            self.rt_prop = rtt_sample
            self._rt_prop_stamp = now

        self._update_bw(bytes_acked, now)
        bdp = self._bdp()

        # --- phase machine ---
        if self.mode == "STARTUP":
            # exit when bandwidth stops growing meaningfully (pipe found)
            if self.btl_bw >= self._full_bw * 1.25 and self.btl_bw > 0:
                self._full_bw = self.btl_bw
                self._full_bw_count = 0
            else:
                self._full_bw_count += 1
            self._cwnd = max(self.CWND_GAIN * self.STARTUP_GAIN * bdp, 4 * self.mss)
            if self._full_bw_count >= 3:
                self.mode = "DRAIN"

        elif self.mode == "DRAIN":
            self._cwnd = max(self.DRAIN_GAIN * self.CWND_GAIN * bdp, 4 * self.mss)
            # once inflight drains to ~BDP, enter steady state
            if self._cwnd <= self.CWND_GAIN * bdp:
                self.mode = "PROBE_BW"
                self._cycle_stamp = now
                self._cycle_idx = 0

        elif self.mode == "PROBE_BW":
            # advance the gain cycle roughly every rt_prop
            if now - self._cycle_stamp >= max(self.rt_prop, 0.05):
                self._cycle_idx = (self._cycle_idx + 1) % len(self.PROBE_BW_GAINS)
                self._cycle_stamp = now
            gain = self.PROBE_BW_GAINS[self._cycle_idx]
            self._cwnd = max(gain * self.CWND_GAIN * bdp, 4 * self.mss)
            # periodically drop into PROBE_RTT to refresh min-RTT
            if now - self._rt_prop_stamp > self.RTPROP_WIN:
                self.mode = "PROBE_RTT"
                self._probe_rtt_done = now + self.PROBE_RTT_DUR

        elif self.mode == "PROBE_RTT":
            self._cwnd = 4 * self.mss    # drain to expose true min-RTT
            if self._probe_rtt_done and now >= self._probe_rtt_done:
                self._rt_prop_stamp = now
                self.mode = "PROBE_BW"
                self._cycle_stamp = now

    def on_loss(self, now):
        # BBR is NOT loss-driven. It ignores loss as a primary signal, but we
        # apply a mild floor so a pathological loss storm can't run away.
        self._cwnd = max(self._cwnd * 0.9, 4 * self.mss)