"""
fairness.py — multi-flow fairness testing (the deployability question).

Every experiment so far ran ONE flow alone on a link. Real networks have MANY
flows sharing bottlenecks. This harness runs TWO flows through ONE shared
bottleneck and measures how they split it.

Key metric — Jain's fairness index:
    J = (sum x_i)^2 / (n * sum x_i^2)
  where x_i is each flow's throughput.
    J = 1.0  -> perfectly fair (equal split)
    J = 0.5  -> maximally unfair with 2 flows (one flow got everything)

Tests:
  * self-fairness : two copies of the SAME controller. Should be fair (~1.0).
  * friendliness  : your PPO agent vs a CUBIC flow. Does it share, starve
                    CUBIC, or get starved? (BBR is famously unfair to CUBIC —
                    included as a reference.)

Design: both flows send through the SAME forward Channel, so their packets
compete for the same queue and capacity — that competition IS congestion. Each
flow has its own sender/controller/RTT tracking and its own return channel.

Run:  python fairness.py
"""

from __future__ import annotations
import numpy as np
import torch

from channel import Channel
from transport import Sender, Receiver
from controllers import AIMD, Cubic, BBR, CongestionController
from rl_env import AgentController
from train_ppo import train
from rl_env import RandomizedTransportEnv


MSS = 1200
LINK_BPS = 20e6      # shared 20 Mbps link (room for two ~10Mb flows)
DELAY_S = 0.02
BUFFER_B = 120_000
SIM_TIME = 12.0
TICK = 0.001
CONTROL_INTERVAL = 0.05


class PPOControllerWrapper(CongestionController):
    """Wraps a trained PPO model as a controller for the fairness sim. Each
    control interval it builds the same 7-dim observation and sets cwnd from the
    model's greedy action — mirroring how AgentController is driven in the env."""

    def __init__(self, model, link_bps, mss=MSS, device="cpu"):
        super().__init__(mss, init_cwnd_packets=10.0)
        self.model = model
        self.device = device
        self._link_pps = link_bps / (mss * 8)
        self._loss_ewma = 0.0
        self.acked_since = 0
        self.lost_since = 0
        self._last_decision = 0.0
        self.sender = None   # set by the harness after sender creation

    def on_ack(self, bytes_acked, rtt_sample, now):
        self.acked_since += bytes_acked

    def on_loss(self, now):
        self.lost_since += 1

    def maybe_decide(self, now):
        """Called each tick; acts every CONTROL_INTERVAL like the env does."""
        if now - self._last_decision < CONTROL_INTERVAL:
            return
        dt = now - self._last_decision if self._last_decision > 0 else CONTROL_INTERVAL
        self._last_decision = now
        s = self.sender
        rtt_min = s.rtt.rtt_min
        srtt = s.rtt.srtt or rtt_min
        if rtt_min == float("inf") or rtt_min == 0:
            rtt_min = DELAY_S * 2; inflation = 1.0
        else:
            inflation = srtt / rtt_min
        thru_pps = (self.acked_since / self.mss) / dt if dt > 0 else 0
        thru_norm = thru_pps / self._link_pps
        loss_rate = self.lost_since / max(1, (self.acked_since / self.mss) + self.lost_since)
        self._loss_ewma = 0.85 * self._loss_ewma + 0.15 * loss_rate
        cwnd_norm = self.cwnd_packets / 100.0
        inflight_ratio = s.bytes_inflight / max(self.cwnd, 1)
        obs = np.array([inflation, rtt_min / 0.1, thru_norm, loss_rate,
                        cwnd_norm, inflight_ratio, self._loss_ewma], dtype=np.float32)
        obs = np.nan_to_num(obs, nan=0.0, posinf=10.0, neginf=-10.0)
        with torch.no_grad():
            a = self.model.actor(torch.as_tensor(obs, device=self.device)).cpu().numpy()
        a = float(np.asarray(a).reshape(-1)[0])
        mult = float(np.clip(0.5 + (a + 1) * 0.5, 0.5, 1.5))
        self._cwnd = max(2 * self.mss, self._cwnd * mult)
        self.acked_since = 0
        self.lost_since = 0


def run_two_flows(ctrl_a, ctrl_b, label_a, label_b, seed=1):
    """Run two flows through one shared bottleneck; return their throughputs."""
    forward = Channel(LINK_BPS, DELAY_S, BUFFER_B, 0.0, seed)
    rev_a = Channel(LINK_BPS, DELAY_S, BUFFER_B, 0.0, seed + 1)
    rev_b = Channel(LINK_BPS, DELAY_S, BUFFER_B, 0.0, seed + 2)

    sender_a = Sender(ctrl_a, forward, rev_a, mss=MSS, total_bytes=10**9)
    sender_b = Sender(ctrl_b, forward, rev_b, mss=MSS, total_bytes=10**9)
    recv_a = Receiver(rev_a, mss=MSS)
    recv_b = Receiver(rev_b, mss=MSS)

    # Route delivered data packets to the right receiver by tagging sender id.
    # Simpo approach: give each flow its own forward-delivery via packet.flow.
    def deliver(pkt, t):
        # ACK packets don't reach here (they go on reverse channels);
        # data packets carry .flow set below.
        if getattr(pkt, "flow", 0) == 0:
            recv_a.on_data_received(pkt, t)
        else:
            recv_b.on_data_received(pkt, t)
    forward.on_deliver = deliver
    rev_a.on_deliver = lambda p, t: sender_a.on_ack_received(p, t)
    rev_b.on_deliver = lambda p, t: sender_b.on_ack_received(p, t)

    # Tag each outgoing data packet with its flow id, so `deliver` routes it to
    # the correct receiver. We set _current_flow before pumping each sender.
    orig_send = forward.send
    _current_flow = {"id": 0}
    def tagged_send(pkt):
        pkt.flow = _current_flow["id"]
        orig_send(pkt)
    forward.send = tagged_send

    def pump_a(now):
        _current_flow["id"] = 0
        Sender.pump(sender_a, now)
    def pump_b(now):
        _current_flow["id"] = 1
        Sender.pump(sender_b, now)

    # attach sender ref for PPO wrappers
    if isinstance(ctrl_a, PPOControllerWrapper): ctrl_a.sender = sender_a
    if isinstance(ctrl_b, PPOControllerWrapper): ctrl_b.sender = sender_b

    now = 0.0
    pump_a(now); pump_b(now)
    while now < SIM_TIME:
        now += TICK
        forward.run_until(now)
        rev_a.run_until(now)
        rev_b.run_until(now)
        sender_a.check_timeouts(now)
        sender_b.check_timeouts(now)
        # PPO controllers make decisions on their interval
        if isinstance(ctrl_a, PPOControllerWrapper): ctrl_a.maybe_decide(now)
        if isinstance(ctrl_b, PPOControllerWrapper): ctrl_b.maybe_decide(now)
        pump_a(now); pump_b(now)

    tput_a = (sender_a.acked_bytes * 8) / 1e6 / now
    tput_b = (sender_b.acked_bytes * 8) / 1e6 / now
    return tput_a, tput_b


def jain(x):
    x = np.array(x, dtype=float)
    return (x.sum() ** 2) / (len(x) * (x ** 2).sum()) if (x ** 2).sum() > 0 else 0.0


def main():
    import os
    from train_and_save import load_agent, SAVE_PATH
    if not os.path.exists(SAVE_PATH):
        print(f"No saved agent found at {SAVE_PATH}.")
        print("Run  python train_and_save.py  first to train & save a good agent.")
        return
    print(f"Loading trained agent from {SAVE_PATH}...")
    model = load_agent(SAVE_PATH)
    print("  loaded.\n")

    def ppo(): 
        c = PPOControllerWrapper(model, LINK_BPS)
        return c

    print("=" * 68)
    print(f"MULTI-FLOW FAIRNESS  |  two flows share a {LINK_BPS/1e6:.0f} Mbps link")
    print("=" * 68)
    print(f"{'matchup':>22} | {'flow A':>8} {'flow B':>8} | {'Jain':>6}")
    print("-" * 68)

    matchups = [
        ("PPO vs PPO",   ppo(),        ppo(),         "PPO", "PPO"),
        ("CUBIC vs CUBIC", Cubic(),    Cubic(),       "CUBIC", "CUBIC"),
        ("PPO vs CUBIC",  ppo(),       Cubic(),       "PPO", "CUBIC"),
        ("BBR vs CUBIC",  BBR(),       Cubic(),       "BBR", "CUBIC"),
        ("PPO vs BBR",    ppo(),       BBR(),         "PPO", "BBR"),
    ]
    for name, ca, cb, la, lb in matchups:
        ta, tb = run_two_flows(ca, cb, la, lb)
        j = jain([ta, tb])
        print(f"{name:>22} | {ta:6.2f}Mb {tb:6.2f}Mb | {j:5.3f}")

    print("-" * 68)
    print("Jain 1.00 = perfectly fair split; 0.50 = one flow starved the other.")
    print("* PPO vs PPO near 1.0 -> your controller is fair to itself.")
    print("* PPO vs CUBIC: is it friendly (near equal) or does one dominate?")
    print("* BBR vs CUBIC is the reference for KNOWN unfairness (BBR starves CUBIC).")


if __name__ == "__main__":
    main()