"""
train_cross.py — train an agent that learns to compete with CUBIC.

Trains on CrossTrafficEnv (some episodes have a competing CUBIC flow), so the
agent experiences being starved and learns to hold its share. Saves to
agent_cross.pt (keeping the original agent.pt for comparison).

After this, run fairness.py pointed at agent_cross.pt (or use compare_fairness.py)
to see if PPO-vs-CUBIC Jain improved from 0.81 toward fair.

Run:  python train_cross.py
(Slower than normal training — cross-traffic episodes run two transports.)
"""

from __future__ import annotations
import numpy as np
import torch

from rl_env import CrossTrafficEnv, TransportEnv
from train_ppo import ActorCritic, train

SAVE_PATH = "agent_cross.pt"
UPDATES = 300
SEED = 0


def eval_goodput(model, seed=999):
    env = TransportEnv(loss_rate=0.0, episode_s=8.0)
    obs = env.reset(seed=seed); gps=[]; done=False
    while not done:
        with torch.no_grad():
            a = model.actor(torch.as_tensor(obs, dtype=torch.float32)).cpu().numpy()
        obs, r, done, info = env.step(a); gps.append(info["goodput_norm"])
    return float(np.mean(gps))


def main():
    print(f"Training cross-traffic-aware agent ({UPDATES} updates)...")
    print("(some episodes include a competing CUBIC flow — slower)\n")
    torch.manual_seed(SEED); np.random.seed(SEED)
    env = CrossTrafficEnv(episode_s=8.0)
    model, _ = train(total_updates=UPDATES, log_every=20, env=env)

    gp = eval_goodput(model)
    print(f"\nGreedy goodput on standard (solo) link: {gp:.2f}x")
    torch.save({"state_dict": model.state_dict(), "obs_dim": env.obs_dim,
                "act_dim": env.act_dim, "goodput": gp}, SAVE_PATH)
    print(f"Saved to {SAVE_PATH}.")
    print("\nNext: compare fairness of agent.pt vs agent_cross.pt to see if the")
    print("CUBIC-sharing improved (PPO-vs-CUBIC Jain should rise above 0.81).")


if __name__ == "__main__":
    main()