"""
generalize.py — the memorize-vs-learn test (the most important experiment now).

Your multi-seed result proved TRAINING is reliable. It did NOT prove the policy
learned general congestion control — every agent trained AND tested on the same
link. This harness answers the open question:

    Did the agent learn a RULE (responds to RTT/loss/throughput signals, works
    on links it never saw) or did it MEMORIZE one link (great on the training
    condition, collapses elsewhere)?

Method:
  1. Train N agents on the STANDARD link (10 Mbps, 20ms, 60KB buffer).
  2. WITHOUT retraining, evaluate each agent on a grid of UNSEEN conditions:
       - different bandwidths (5, 20, 40 Mbps)
       - different RTTs (10ms, 50ms, 100ms one-way)
       - different buffers (small, large)
       - added random loss (1%, 3%)
  3. Run CUBIC and AIMD on each condition too, for comparison.
  4. Report goodput / latency-inflation / loss per condition.

Reading it:
  * Agent holds up across conditions  -> it GENERALIZED. Real congestion control.
  * Agent collapses off the training link -> it MEMORIZED. Known weakness of
    learned CC; the fix is to TRAIN across varied conditions (domain
    randomization). Either result is publishable/thesis-worthy if reported
    honestly.

Run:  python generalize.py
(Trains N agents once, then evaluation is fast — no retraining per condition.)
"""

from __future__ import annotations
import numpy as np
import torch

from rl_env import TransportEnv
from train_ppo import ActorCritic, train
from simulate import run as run_baseline
from controllers import AIMD, Cubic


# --- how many agents to train (more = more robust averages, slower) ---
TRAIN_SEEDS = [0, 1, 2]        # 3 agents; bump to 5 for the final run
TRAIN_UPDATES = 200

# The STANDARD training link (what agents are trained on).
TRAIN_CFG = dict(bottleneck_bps=10e6, delay_s=0.02, buffer_bytes=60_000, loss_rate=0.0)

# The grid of conditions to TEST on. The first entry is the training condition
# itself (sanity check — should match your multi-seed numbers).
TEST_CONDITIONS = [
    # name,                bw(bps), delay(s), buffer(B), loss
    ("TRAIN 10Mb/20ms",     10e6,   0.02,     60_000,    0.0),   # seen (baseline)
    ("bw  5Mb (unseen)",     5e6,   0.02,     60_000,    0.0),
    ("bw 20Mb (unseen)",    20e6,   0.02,     60_000,    0.0),
    ("bw 40Mb (unseen)",    40e6,   0.02,     60_000,    0.0),
    ("rtt 10ms (unseen)",   10e6,   0.005,    60_000,    0.0),
    ("rtt 50ms (unseen)",   10e6,   0.05,     60_000,    0.0),
    ("rtt 100ms(unseen)",   10e6,   0.10,     60_000,    0.0),
    ("buf small(unseen)",   10e6,   0.02,     20_000,    0.0),
    ("buf large(unseen)",   10e6,   0.02,     200_000,   0.0),
    ("loss 1%  (unseen)",   10e6,   0.02,     60_000,    0.01),
    ("loss 3%  (unseen)",   10e6,   0.02,     60_000,    0.03),
]

EVAL_SEED = 999


def eval_agent_on(model, bw, delay, buf, loss, device="cpu"):
    """Greedy-evaluate a trained agent on ONE condition -> (goodput, infl, loss)."""
    env = TransportEnv(bottleneck_bps=bw, delay_s=delay, buffer_bytes=buf,
                       loss_rate=loss, episode_s=8.0)
    obs = env.reset(seed=EVAL_SEED)
    infos = []
    done = False
    while not done:
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device)
        with torch.no_grad():
            action = model.actor(obs_t).cpu().numpy()   # greedy
        obs, r, done, info = env.step(action)
        infos.append(info)
    gp = float(np.mean([i["goodput_norm"] for i in infos]))
    infl = float(np.mean([i["inflation"] for i in infos]))
    lo = float(np.mean([i["loss"] for i in infos]))
    return gp, infl, lo


def eval_baseline_on(ctrl_factory, bw, delay, buf, loss):
    """Run a classic controller on one condition -> (goodput_frac, infl, loss)."""
    ctrl = ctrl_factory()
    r = run_baseline(ctrl, bottleneck_bps=bw, delay_s=delay, buffer_bytes=buf,
                     loss_rate=loss, total_bytes=4_000_000)
    gp = r["goodput_mbps"] / (bw / 1e6)          # fraction of THIS link's capacity
    infl = (r["srtt_ms"] / r["rtt_min_ms"]) if r["rtt_min_ms"] > 0 else 0.0
    lo = r["loss_pct"] / 100.0
    return gp, infl, lo


def main():
    print("=" * 78)
    print("GENERALIZATION TEST  |  train on standard link, test on unseen conditions")
    print("=" * 78)
    print(f"Training {len(TRAIN_SEEDS)} agents on {TRAIN_CFG['bottleneck_bps']/1e6:.0f}Mb"
          f"/{TRAIN_CFG['delay_s']*1000:.0f}ms... (one-time)\n")

    # Train the agents on the standard link.
    models = []
    for seed in TRAIN_SEEDS:
        torch.manual_seed(seed); np.random.seed(seed)
        # temporarily point the trainer's env at the standard cfg via module default
        model, _ = train(total_updates=200, log_every=1)
        models.append(model)
        print(f"  agent seed {seed} trained.")
    print()

    # Evaluate every agent on every condition; average across agents.
    print(f"{'condition':>18} | {'PPO gp':>7} {'PPO lat':>7} {'PPO ls':>6} | "
          f"{'CUB gp':>6} {'CUB lat':>7} | {'AIM gp':>6} {'AIM lat':>7}")
    print("-" * 78)

    summary = []
    for name, bw, delay, buf, loss in TEST_CONDITIONS:
        # PPO: average over the trained agents
        gps, infls, losses = [], [], []
        for m in models:
            gp, infl, lo = eval_agent_on(m, bw, delay, buf, loss)
            gps.append(gp); infls.append(infl); losses.append(lo)
        pgp, pinfl, plo = np.mean(gps), np.mean(infls), np.mean(losses)

        cgp, cinfl, _ = eval_baseline_on(Cubic, bw, delay, buf, loss)
        agp, ainfl, _ = eval_baseline_on(AIMD, bw, delay, buf, loss)

        print(f"{name:>18} | {pgp:6.2f}x {pinfl:6.2f}x {plo*100:5.1f}% | "
              f"{cgp:5.2f}x {cinfl:6.2f}x | {agp:5.2f}x {ainfl:6.2f}x")
        summary.append((name, pgp, pinfl, plo, cgp, cinfl, agp, ainfl))

    # --- verdict ---
    print("\n" + "=" * 78)
    print("HOW TO READ THIS")
    print("=" * 78)
    train_gp = summary[0][1]
    unseen = summary[1:]
    worst = min(unseen, key=lambda s: s[1])
    print(f"* Training-link goodput: {train_gp:.2f}x (should match your multi-seed run).")
    print(f"* Worst unseen goodput : {worst[1]:.2f}x  (condition: {worst[0].strip()}).")
    print(f"* If unseen goodput stays high & latency stays low across the grid")
    print(f"  -> the policy GENERALIZED. It learned a rule, not a lookup table.")
    print(f"* If goodput craters or latency explodes on some conditions")
    print(f"  -> it partly MEMORIZED. Fix: train across VARIED conditions")
    print(f"     (domain randomization). This is itself an honest, reportable finding.")
    print(f"* Compare each cell to CUBIC/AIMD on the SAME condition — beating them")
    print(f"  on UNSEEN links is much stronger evidence than on the training link.")


if __name__ == "__main__":
    main()