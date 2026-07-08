"""
train_and_save.py — train ONE good agent and save it to disk.

The problem: every harness (fairness.py, generalize.py, compare_all.py) trained a
FRESH agent inline, and those came out inconsistent — sometimes weak (0.15-0.40x
goodput), making results untrustworthy. The fix: train one good agent ONCE, save
it, and have every harness LOAD it. Consistent agent -> trustworthy comparisons.

This trains on the randomized env (so the saved agent is a generalist), for more
updates than the default so it reliably reaches full strength, then saves weights
to  agent.pt  and reports the trained goodput so you can confirm it's good BEFORE
trusting any downstream experiment.

Run:  python train_and_save.py
Then re-run fairness.py / compare_all.py — they'll load agent.pt.
"""

from __future__ import annotations
import numpy as np
import torch

from rl_env import RandomizedTransportEnv, TransportEnv
from train_ppo import ActorCritic, train

SAVE_PATH = "agent.pt"
UPDATES = 400          # more than the 200 default, to reliably reach full strength
SEED = 0


def eval_goodput(model, device="cpu", seed=999):
    """Greedy eval on the standard link -> goodput (sanity that it trained well)."""
    env = TransportEnv(loss_rate=0.0, episode_s=8.0)
    obs = env.reset(seed=seed)
    gps = []
    done = False
    while not done:
        with torch.no_grad():
            a = model.actor(torch.as_tensor(obs, dtype=torch.float32)).cpu().numpy()
        obs, r, done, info = env.step(a)
        gps.append(info["goodput_norm"])
    return float(np.mean(gps))


def main():
    print(f"Training one good agent ({UPDATES} updates, randomized env)...")
    torch.manual_seed(SEED); np.random.seed(SEED)
    env = RandomizedTransportEnv(episode_s=8.0)
    model, _ = train(total_updates=UPDATES, log_every=20, env=env)

    gp = eval_goodput(model)
    print(f"\nTrained agent greedy goodput on standard link: {gp:.2f}x")
    if gp < 0.7:
        print("WARNING: goodput < 0.7x — agent under-trained. Consider more updates")
        print("or check MU in rl_env.py (should be 2.0). Saving anyway.")
    else:
        print("Good — agent trained to full strength.")

    # Save obs_dim alongside weights so loaders can rebuild the right architecture.
    torch.save({
        "state_dict": model.state_dict(),
        "obs_dim": env.obs_dim,
        "act_dim": env.act_dim,
        "goodput": gp,
    }, SAVE_PATH)
    print(f"Saved to {SAVE_PATH}. Downstream harnesses can now load this agent.")


def load_agent(path=SAVE_PATH, device="cpu"):
    """Helper other scripts import to load the saved agent."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = ActorCritic(ckpt["obs_dim"], ckpt["act_dim"]).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model


if __name__ == "__main__":
    main()