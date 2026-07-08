"""
rl_env.py — the transport sim as a reinforcement-learning environment (items 14-15).

This is the bridge between your protocol and any RL agent. It follows the
standard Gym/Gymnasium contract (reset -> obs; step(action) -> obs, reward,
done, info) but has NO hard dependency on gymnasium, so it runs anywhere.

THE MDP:

  State (what the agent observes each control interval):
      [ srtt/rtt_min,        # latency inflation (1.0 = no queue, >1 = bufferbloat)
        rtt_min_norm,        # the path's base RTT (context)
        throughput_norm,     # recent delivery rate / link estimate
        loss_rate,           # recent loss fraction
        cwnd_norm,           # current window (normalized)
        inflight_ratio ]     # inflight / cwnd  (how full the pipe is)

  Action (what the agent controls):
      a continuous multiplier on cwnd in [0.5, 1.5] (clamped). The agent nudges
      the window up or down each interval. (Discrete variant is trivial to add.)

  Reward (the PCC utility — the heart of the research):
      r = throughput  -  LAMBDA * latency_inflation  -  MU * loss
      Tune LAMBDA / MU to encode priorities. High LAMBDA -> latency-sensitive
      (the game-traffic regime). This weighting is itself a contribution.

The agent acts every CONTROL_INTERVAL seconds of sim time; between decisions the
protocol runs normally with the window the agent set.
"""

from __future__ import annotations
import numpy as np
from channel import Channel
from transport import Sender, Receiver
from controllers import CongestionController


class AgentController(CongestionController):
    """A controller whose window is set directly by the RL agent, not by a
    fixed rule. Implements the SAME interface as AIMD, so the transport code is
    unchanged. on_ack/on_loss just track stats; the agent sets cwnd in step()."""

    def __init__(self, mss=1200, init_cwnd_packets=10.0):
        super().__init__(mss, init_cwnd_packets)
        self.acked_since = 0
        self.lost_since = 0

    def on_ack(self, bytes_acked, rtt_sample, now):
        self.acked_since += bytes_acked

    def on_loss(self, now):
        self.lost_since += 1

    def set_cwnd(self, cwnd_bytes):
        self._cwnd = max(2 * self.mss, cwnd_bytes)


class TransportEnv:
    LAMBDA = 1.0      # latency penalty weight
    MU = 4.0          # loss penalty weight
    CONTROL_INTERVAL = 0.05   # agent decides every 50 ms of sim time

    def __init__(self, bottleneck_bps=10e6, delay_s=0.02, buffer_bytes=60_000,
                 loss_rate=0.0, mss=1200, episode_s=10.0, seed=1):
        self.cfg = dict(bottleneck_bps=bottleneck_bps, delay_s=delay_s,
                        buffer_bytes=buffer_bytes, loss_rate=loss_rate,
                        mss=mss, episode_s=episode_s)
        self.mss = mss
        self.seed = seed
        self.obs_dim = 7   # added smoothed loss-history feature
        self.act_dim = 1

    def reset(self, seed=None):
        sampled = self._sample_conditions()
        if sampled:
            self.cfg.update(sampled)
        c = self.cfg
        s = self.seed if seed is None else seed
        self.forward = Channel(c["bottleneck_bps"], c["delay_s"],
                               c["buffer_bytes"], c["loss_rate"], s)
        self.reverse = Channel(c["bottleneck_bps"], c["delay_s"],
                               c["buffer_bytes"], 0.0, s + 1)
        self.cc = AgentController(mss=self.mss, init_cwnd_packets=10.0)
        # large total so the flow never "finishes" before the episode ends
        self.sender = Sender(self.cc, self.forward, self.reverse,
                             mss=self.mss, total_bytes=10**9)
        self.receiver = Receiver(self.reverse, mss=self.mss)
        self.forward.on_deliver = lambda p, t: self.receiver.on_data_received(p, t)
        self.reverse.on_deliver = lambda p, t: self.sender.on_ack_received(p, t)

        self.now = 0.0
        self.tick = 0.001
        self._link_pps = c["bottleneck_bps"] / (self.mss * 8)  # packets/s capacity
        self._loss_ewma = 0.0   # smoothed loss rate, so agent perceives SUSTAINED loss
        self.sender.pump(self.now)
        return self._observe(acked=0, lost=0, dt=self.CONTROL_INTERVAL)

    def _observe(self, acked, lost, dt):
        rtt_min = self.sender.rtt.rtt_min
        srtt = self.sender.rtt.srtt or rtt_min
        if rtt_min == float("inf") or rtt_min == 0:
            rtt_min = self.cfg["delay_s"] * 2
            inflation = 1.0
        else:
            inflation = srtt / rtt_min
        thru_pps = (acked / self.mss) / dt if dt > 0 else 0
        thru_norm = thru_pps / self._link_pps
        loss_rate = lost / max(1, (acked / self.mss) + lost)
        # smoothed loss (EWMA) so the agent can distinguish a chronically lossy
        # link from a one-off spike -- the key signal it lacked before.
        self._loss_ewma = 0.85 * self._loss_ewma + 0.15 * loss_rate
        cwnd_norm = self.cc.cwnd_packets / 100.0
        inflight_ratio = (self.sender.bytes_inflight / max(self.cc.cwnd, 1))
        obs = np.array([
            inflation, rtt_min / 0.1, thru_norm, loss_rate,
            cwnd_norm, inflight_ratio, self._loss_ewma
        ], dtype=np.float32)
        # Guard: a degenerate (very timid) policy can produce edge-case values;
        # never let NaN/inf reach the network (it crashes torch's Normal()).
        obs = np.nan_to_num(obs, nan=0.0, posinf=10.0, neginf=-10.0)
        return obs

    # allow subclasses to resample link conditions each episode
    def _sample_conditions(self):
        return None  # base env: fixed conditions (use self.cfg unchanged)

    def step(self, action):
        """action: scalar multiplier (we map agent output -> [0.5, 1.5])."""
        a = float(np.asarray(action).reshape(-1)[0])               # scalar action
        mult = float(np.clip(0.5 + (a + 1) * 0.5, 0.5, 1.5))       # action in ~[-1,1]
        self.cc.set_cwnd(self.cc.cwnd * mult)

        self.cc.acked_since = 0
        self.cc.lost_since = 0
        target = self.now + self.CONTROL_INTERVAL
        while self.now < target:
            self.now += self.tick
            self.forward.run_until(self.now)
            self.reverse.run_until(self.now)
            self.sender.check_timeouts(self.now)
            self.sender.pump(self.now)

        acked = self.cc.acked_since
        lost = self.cc.lost_since
        dt = self.CONTROL_INTERVAL
        obs = self._observe(acked, lost, dt)

        # --- PCC utility reward ---
        thru_norm = obs[2]                 # in [0, ~1+]
        inflation = obs[0]                 # >= 1
        loss_rate = obs[3]
        reward = thru_norm - self.LAMBDA * (inflation - 1.0) - self.MU * loss_rate

        done = self.now >= self.cfg["episode_s"]
        info = {"goodput_norm": thru_norm, "inflation": inflation,
                "loss": loss_rate, "cwnd_pkts": self.cc.cwnd_packets,
                "queue": self.forward.queue_bytes}
        return obs, float(reward), done, info


class RandomizedTransportEnv(TransportEnv):
    """Domain randomization: each episode samples a fresh link from wide ranges.

    This directly attacks the generalization gaps the fixed-link agent showed:
      * varied BANDWIDTH  -> agent must learn to scale up/down, not memorize 10Mb
      * varied LOSS       -> agent must learn that loss means back off (the fixed
                             agent, trained at 0% loss, catastrophically overshot
                             on lossy links)
      * varied DELAY      -> agent must handle different RTT regimes

    Ranges are chosen to span (and exceed) the generalization test grid, so the
    'unseen' test conditions become closer to 'seen distribution' — the whole
    point of domain randomization. Sampled log-uniformly for bandwidth so small
    and large links are equally represented.
    """
    import numpy as _np

    BW_RANGE = (3e6, 48e6)      # 3 - 48 Mbps
    DELAY_RANGE = (0.005, 0.06) # 5 - 60 ms one-way
    LOSS_CHOICES = (0.0, 0.0, 0.005, 0.01, 0.02, 0.03, 0.04)  # incl. heavier loss

    def _sample_conditions(self):
        import numpy as np
        lo, hi = self.BW_RANGE
        bw = float(np.exp(np.random.uniform(np.log(lo), np.log(hi))))  # log-uniform
        d0, d1 = self.DELAY_RANGE
        delay = float(np.random.uniform(d0, d1))
        loss = float(np.random.choice(self.LOSS_CHOICES))
        return dict(bottleneck_bps=bw, delay_s=delay, loss_rate=loss)


class CrossTrafficEnv(RandomizedTransportEnv):
    """Domain randomization + sometimes a COMPETING CUBIC flow on the same link.

    Fixes the CUBIC-starvation found in fairness testing: the agent never trained
    with cross-traffic, so it never learned to hold its share against an
    aggressive loss-based flow. Here, on a fraction of episodes, a CUBIC flow
    shares the SAME bottleneck. When CUBIC starves the agent, the agent's
    throughput (and reward) drop -> it learns that excessive politeness costs it,
    and adapts to defend its share.

    Mix: with prob CROSS_PROB an episode has a CUBIC competitor; otherwise solo.
    Keeping solo episodes preserves the skills it already has.
    """
    import numpy as _np
    CROSS_PROB = 0.5     # fraction of episodes that include a CUBIC competitor

    def reset(self, seed=None):
        obs = super().reset(seed=seed)
        import numpy as np
        from controllers import Cubic
        from transport import Sender, Receiver
        from channel import Channel
        self._has_cross = (np.random.random() < self.CROSS_PROB)
        if self._has_cross:
            c = self.cfg
            s = (self.seed if seed is None else seed) + 100
            # CUBIC competitor shares the SAME forward channel (competes for buffer)
            self._x_reverse = Channel(c["bottleneck_bps"], c["delay_s"],
                                      c["buffer_bytes"], 0.0, s + 3)
            self._x_cc = Cubic(mss=self.mss, init_cwnd_packets=2)
            self._x_sender = Sender(self._x_cc, self.forward, self._x_reverse,
                                    mss=self.mss, total_bytes=10**9)
            self._x_receiver = Receiver(self._x_reverse, mss=self.mss)
            # route: agent packets (flow 0) to agent receiver, CUBIC (flow 1) to its own
            agent_recv = self.receiver
            x_recv = self._x_receiver
            def deliver(pkt, t):
                if getattr(pkt, "flow", 0) == 0:
                    agent_recv.on_data_received(pkt, t)
                else:
                    x_recv.on_data_received(pkt, t)
            self.forward.on_deliver = deliver
            self._x_reverse.on_deliver = lambda p, t: self._x_sender.on_ack_received(p, t)
            # tag packets by flow on send
            orig_send = self.forward.send
            self._cur_flow = {"id": 0}
            def tagged(pkt):
                pkt.flow = self._cur_flow["id"]
                orig_send(pkt)
            self.forward.send = tagged
            self._cur_flow["id"] = 1
            self._x_sender.pump(self.now)
            self._cur_flow["id"] = 0
        return obs

    def step(self, action):
        # Agent acts as usual, but if there's cross-traffic, pump CUBIC too.
        if not getattr(self, "_has_cross", False):
            return super().step(action)

        import numpy as np
        a = float(np.asarray(action).reshape(-1)[0])
        mult = float(np.clip(0.5 + (a + 1) * 0.5, 0.5, 1.5))
        self.cc.set_cwnd(self.cc.cwnd * mult)
        self.cc.acked_since = 0
        self.cc.lost_since = 0
        target = self.now + self.CONTROL_INTERVAL
        while self.now < target:
            self.now += self.tick
            self.forward.run_until(self.now)
            self.reverse.run_until(self.now)
            self._x_reverse.run_until(self.now)
            # agent flow
            self._cur_flow["id"] = 0
            self.sender.check_timeouts(self.now)
            self.sender.pump(self.now)
            # cubic competitor flow
            self._cur_flow["id"] = 1
            self._x_sender.check_timeouts(self.now)
            self._x_sender.pump(self.now)
            self._cur_flow["id"] = 0

        acked = self.cc.acked_since
        lost = self.cc.lost_since
        dt = self.CONTROL_INTERVAL
        obs = self._observe(acked, lost, dt)
        thru_norm = obs[2]; inflation = obs[0]; loss_rate = obs[3]
        reward = thru_norm - self.LAMBDA * (inflation - 1.0) - self.MU * loss_rate
        done = self.now >= self.cfg["episode_s"]
        info = {"goodput_norm": thru_norm, "inflation": inflation,
                "loss": loss_rate, "cwnd_pkts": self.cc.cwnd_packets,
                "queue": self.forward.queue_bytes}
        return obs, float(reward), done, info