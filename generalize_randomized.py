"""
generalize_randomized.py — does domain randomization close the generalization gaps?

The fixed-link agent (generalize.py) memorized its training regime: it couldn't
scale to fast links and it catastrophically overshot on lossy links (24-32% loss)
because it never saw loss in training.

This trains agents on the RANDOMIZED env (varied bw/delay/loss every episode) and
runs the SAME generalization grid. Compare the two tables:
  * If loss columns drop from ~24-32% to single digits, and fast-link goodput
    rises -> domain randomization worked. The agent learned a RULE.
  * The training-link number may dip slightly (a generalist is rarely as good on
    one specific link as a specialist) — that's the expected, honest tradeoff.

Run:  python generalize_randomized.py
"""

from __future__ import annotations
import numpy as np
import torch

from rl_env import RandomizedTransportEnv
from train_ppo import ActorCritic, train
# reuse the exact eval helpers + condition grid from the fixed-link harness
from generalize import (eval_agent_on, eval_baseline_on, TEST_CONDITIONS,
                        TRAIN_SEEDS, TRAIN_UPDATES)
from controllers import AIMD, Cubic


def main():
    print("=" * 78)
    print("DOMAIN-RANDOMIZED TRAINING  |  varied bw/delay/loss each episode")
    print("=" * 78)
    print(f"Training {len(TRAIN_SEEDS)} agents on RANDOMIZED links... (one-time)\n")

    models = []
    for seed in TRAIN_SEEDS:
        torch.manual_seed(seed); np.random.seed(seed)
        env = RandomizedTransportEnv(episode_s=8.0)
        model, _ = train(total_updates=TRAIN_UPDATES, log_every=1, env=env)
        models.append(model)
        print(f"  agent seed {seed} trained (randomized).")
    print()

    print(f"{'condition':>18} | {'PPO gp':>7} {'PPO lat':>7} {'PPO ls':>6} | "
          f"{'CUB gp':>6} {'CUB lat':>7} | {'AIM gp':>6} {'AIM lat':>7}")
    print("-" * 78)

    for name, bw, delay, buf, loss in TEST_CONDITIONS:
        gps, infls, losses = [], [], []
        for m in models:
            gp, infl, lo = eval_agent_on(m, bw, delay, buf, loss)
            gps.append(gp); infls.append(infl); losses.append(lo)
        pgp, pinfl, plo = np.mean(gps), np.mean(infls), np.mean(losses)
        cgp, cinfl, _ = eval_baseline_on(Cubic, bw, delay, buf, loss)
        agp, ainfl, _ = eval_baseline_on(AIMD, bw, delay, buf, loss)
        print(f"{name:>18} | {pgp:6.2f}x {pinfl:6.2f}x {plo*100:5.1f}% | "
              f"{cgp:5.2f}x {cinfl:6.2f}x | {agp:5.2f}x {ainfl:6.2f}x")

    print("\n" + "=" * 78)
    print("COMPARE TO generalize.py's table:")
    print("* Did the loss 1% / loss 3% columns drop from ~24-32% to single digits?")
    print("* Did bw 20Mb / 40Mb goodput rise (better scaling to fast links)?")
    print("* Training-link (10Mb/20ms) may dip a little — the generalist tradeoff.")
    print("If loss handling improved, you've shown diagnose -> fix. That's the")
    print("core research loop, and a strong thesis result.")


if __name__ == "__main__":
    main()