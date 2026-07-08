"""
compare_all.py — the full three-way comparison (PPO vs CUBIC vs BBR vs AIMD).

Shows goodput / latency / LOSS for every controller on every condition, so the
real story is visible:
  * CUBIC  trades LATENCY for throughput (bufferbloat).
  * BBR    trades LOSS for throughput+latency (ignores loss, over-sends).
  * AIMD   is the weak classic baseline.
  * PPO (domain-randomized) aims for good throughput AND low latency AND low loss.

Trains randomized agents (generalists), then evaluates across the grid.
Run:  python compare_all.py
"""

from __future__ import annotations
import numpy as np
import torch

from rl_env import RandomizedTransportEnv
from train_ppo import train
from simulate import run as run_baseline
from generalize import eval_agent_on, TEST_CONDITIONS, TRAIN_SEEDS, TRAIN_UPDATES
from controllers import AIMD, Cubic, BBR


def baseline_on(ctrl_factory, bw, delay, buf, loss):
    r = run_baseline(ctrl_factory(), bottleneck_bps=bw, delay_s=delay,
                     buffer_bytes=buf, loss_rate=loss, total_bytes=4_000_000)
    gp = r["goodput_mbps"] / (bw / 1e6)
    infl = (r["srtt_ms"] / r["rtt_min_ms"]) if r["rtt_min_ms"] > 0 else 0.0
    lo = r["loss_pct"] / 100.0
    return gp, infl, lo


def main():
    print("=" * 92)
    print("FULL COMPARISON  |  domain-randomized PPO vs CUBIC vs BBR vs AIMD")
    print("=" * 92)
    print(f"Training {len(TRAIN_SEEDS)} randomized agents...\n")

    models = []
    for seed in TRAIN_SEEDS:
        torch.manual_seed(seed); np.random.seed(seed)
        env = RandomizedTransportEnv(episode_s=8.0)
        model, _ = train(total_updates=TRAIN_UPDATES, log_every=1, env=env)
        models.append(model)
        print(f"  agent seed {seed} trained.")
    print()

    hdr = (f"{'condition':>17} | {'PPO':>17} | {'CUBIC':>17} | "
           f"{'BBR':>17} | {'AIMD':>17}")
    print(hdr)
    print(f"{'':>17} | {'gp   lat   loss':>17} | {'gp   lat   loss':>17} | "
          f"{'gp   lat   loss':>17} | {'gp   lat   loss':>17}")
    print("-" * 92)

    def cell(gp, infl, lo):
        return f"{gp:4.2f} {infl:4.2f} {lo*100:4.1f}%"

    for name, bw, delay, buf, loss in TEST_CONDITIONS:
        gps, infls, losses = [], [], []
        for m in models:
            gp, infl, lo = eval_agent_on(m, bw, delay, buf, loss)
            gps.append(gp); infls.append(infl); losses.append(lo)
        p = (np.mean(gps), np.mean(infls), np.mean(losses))
        c = baseline_on(Cubic, bw, delay, buf, loss)
        b = baseline_on(BBR, bw, delay, buf, loss)
        a = baseline_on(AIMD, bw, delay, buf, loss)
        print(f"{name:>17} | {cell(*p)} | {cell(*c)} | {cell(*b)} | {cell(*a)}")

    print("\n" + "=" * 92)
    print("READING IT (each cell: goodput  latency-inflation  loss):")
    print("* PPO's pitch: competitive goodput, low latency, low loss — all three at once.")
    print("* Watch BBR's LOSS column: often high (it ignores loss) — that's its real cost.")
    print("* Watch CUBIC's LATENCY column: often high (bufferbloat) — that's its cost.")
    print("* Where PPO matches their goodput while beating BOTH costs, that's the result.")


if __name__ == "__main__":
    main()