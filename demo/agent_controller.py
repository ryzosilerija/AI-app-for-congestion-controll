"""
agent_controller.py — wraps the REAL trained PPO agent (agent.pt) as a
window-based congestion controller for the live demo.

The agent outputs a window multiplier (same as training), which directly
sets cwnd. The server gates sends on bytes_inflight < cwnd, so the agent
has real leverage over latency — exactly the mechanism it was trained on.
"""
from __future__ import annotations
import os
import time
import numpy as np

AGENT_PATH = os.path.join(os.path.dirname(__file__), "..", "agent.pt")


def load_agent():
    try:
        import torch
        ckpt = torch.load(AGENT_PATH, map_location="cpu", weights_only=False)
        import sys
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
        from train_ppo import ActorCritic
        model = ActorCritic(ckpt["obs_dim"], ckpt["act_dim"])
        model.load_state_dict(ckpt["state_dict"])
        model.eval()
        print(f"agent_controller: loaded agent.pt  "
              f"(obs_dim={ckpt['obs_dim']} goodput={ckpt.get('goodput',0):.2f}x)")
        return model
    except Exception as e:
        print(f"agent_controller: WARNING could not load agent.pt: {e}")
        return None


class AgentCC:
    """The real trained PPO agent driving cwnd."""
    name = "PPO AGENT (trained RL)"
    CONTROL_INTERVAL = 0.05
    MSS = 1200

    def __init__(self, model):
        self.model = model
        self.cwnd = 20_000          # bytes
        self._cwnd_pkts = 10.0      # internal tracking in packets
        self._loss_ewma = 0.0
        self._last_decision = 0.0
        self._link_pps = 10e6 / (self.MSS * 8)

    def update(self, srtt, rtt_min, loss_events, bytes_acked, **kw):
        now = time.monotonic()
        if now - self._last_decision < self.CONTROL_INTERVAL:
            return
        self._last_decision = now

        if rtt_min <= 0 or srtt <= 0:
            inflation = 1.0
            rtt_min = max(rtt_min, 0.001)
        else:
            inflation = srtt / rtt_min

        thru_pps = (bytes_acked / self.MSS) / self.CONTROL_INTERVAL
        thru_norm = thru_pps / self._link_pps

        loss_rate = min(1.0, loss_events / max(1, bytes_acked / self.MSS + loss_events))
        self._loss_ewma = 0.85 * self._loss_ewma + 0.15 * loss_rate

        cwnd_norm = self._cwnd_pkts / 100.0
        inflight_bytes = kw.get("bytes_inflight", self.cwnd)
        inflight_ratio = inflight_bytes / max(1.0, self._cwnd_pkts * self.MSS)

        obs = np.array([
            inflation, rtt_min / 0.1, thru_norm, loss_rate,
            cwnd_norm, inflight_ratio, self._loss_ewma,
        ], dtype=np.float32)
        obs = np.clip(obs, -10.0, 10.0)

        try:
            import torch
            with torch.no_grad():
                action = self.model.actor(
                    torch.as_tensor(obs, dtype=torch.float32)
                ).numpy()
            a = float(np.asarray(action).reshape(-1)[0])
            mult = float(np.clip(0.5 + (a + 1) * 0.5, 0.5, 1.5))
            self._cwnd_pkts = max(4.0, self._cwnd_pkts * mult)
            self.cwnd = max(5000, int(self._cwnd_pkts * self.MSS))
        except Exception:
            pass