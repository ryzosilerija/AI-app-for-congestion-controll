"""
validate_multiseed.py — the honesty check (the step this whole project needed).

A single good RL run can be luck: a fortunate initialization + exploration path.
A RESULT is something that reproduces across seeds. This harness answers the one
question that separates "nice run" from "real result":

    Does the PPO controller's win over CUBIC/AIMD hold across different seeds,
    or did we get lucky once?

For each seed it:
  1. trains a FRESH PPO agent from scratch (new random init, new exploration),
  2. evaluates it greedily on a held-out eval seed,
  3. runs CUBIC and AIMD through the IDENTICAL environment,
  4. records goodput / latency-inflation / loss for all three.

Then it prints per-seed rows and mean +/- std. If PPO's numbers are tight across
seeds and consistently beat the baselines, you have a real (if narrow) result.
If they're all over the place, your setup needs stabilizing BEFORE you build on it
-- which is itself a hugely valuable thing to learn early.

Run:  python validate_multiseed.py
(Trains N agents, so it takes a while -- reduce SEEDS or updates to go faster.)
"""

from __future__ import annotations
import numpy as np
import torch

from rl_env import TransportEnv
from train_ppo import ActorCritic, train, evaluate  # reuse the trainer
from simulate import run as run_baseline
from controllers import AIMD, Cubic


# --- knobs -------------------------------------------------------------
SEEDS = [0, 1, 2, 3, 4]          # the seeds we validate across
UPDATES = 200                    # PPO updates per agent (same as train_ppo default)
EVAL_SEED = 999                  # held-out seed for evaluation (same for all)


def eval_ppo(model, device="cpu", eval_seed=EVAL_SEED):
    """Greedy evaluation of a trained model -> (goodput, latency_infl, loss)."""
    env = TransportEnv(loss_rate=0.0, episode_s=8.0)
    obs = env.reset(seed=eval_seed)
    infos = []
    done = False
    while not done:
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device)
        with torch.no_grad():
            action = model.actor(obs_t).cpu().numpy()   # greedy = mean action
        obs, r, done, info = env.step(action)
        infos.append(info)
    gp = float(np.mean([i["goodput_norm"] for i in infos]))
    infl = float(np.mean([i["inflation"] for i in infos]))
    loss = float(np.mean([i["loss"] for i in infos]))
    return gp, infl, loss


def eval_baseline(ctrl_factory):
    """Run a classic controller through the same link the env models.
    Returns (goodput_fraction, latency_infl, loss_fraction)."""
    ctrl = ctrl_factory()
    r = run_baseline(ctrl, loss_rate=0.0, total_bytes=4_000_000,
                     bottleneck_bps=10e6, delay_s=0.02, buffer_bytes=60_000)
    # goodput as fraction of link capacity (10 Mbps)
    gp = r["goodput_mbps"] / 10.0
    infl = (r["srtt_ms"] / r["rtt_min_ms"]) if r["rtt_min_ms"] > 0 else 0.0
    loss = r["loss_pct"] / 100.0
    return gp, infl, loss


def main():
    print("=" * 74)
    print(f"MULTI-SEED VALIDATION  |  {len(SEEDS)} seeds x {UPDATES} PPO updates each")
    print("=" * 74)
    print("Training a fresh agent per seed. This takes a while; grab a coffee.\n")

    ppo_results = []
    for seed in SEEDS:
        print(f"--- seed {seed}: training fresh PPO agent ---")
        torch.manual_seed(seed)
        np.random.seed(seed)
        model, _ = train(total_updates=UPDATES, log_every=1)
        gp, infl, loss = eval_ppo(model)
        ppo_results.append((seed, gp, infl, loss))
        print(f"    seed {seed} PPO: goodput {gp:.2f}x | lat-infl {infl:.2f}x | loss {loss*100:.2f}%\n")

    # baselines are deterministic on a fixed link; run once
    cub = eval_baseline(Cubic)
    aim = eval_baseline(AIMD)

    # --- report ---
    print("\n" + "=" * 74)
    print("RESULTS (single-flow, 10 Mbps / 20ms link, 60KB buffer)")
    print("=" * 74)
    print(f"{'seed':>5} | {'goodput':>9} | {'lat-infl':>9} | {'loss':>7}")
    print("-" * 40)
    for seed, gp, infl, loss in ppo_results:
        print(f"{seed:>5} | {gp:8.2f}x | {infl:8.2f}x | {loss*100:6.2f}%")
    print("-" * 40)

    gps = np.array([r[1] for r in ppo_results])
    infls = np.array([r[2] for r in ppo_results])
    losses = np.array([r[3] for r in ppo_results])

    print(f"\nPPO  mean +/- std:")
    print(f"  goodput   : {gps.mean():.3f}x  +/- {gps.std():.3f}")
    print(f"  lat-infl  : {infls.mean():.3f}x +/- {infls.std():.3f}")
    print(f"  loss      : {losses.mean()*100:.2f}% +/- {losses.std()*100:.2f}%")

    print(f"\nBaselines (same link):")
    print(f"  CUBIC     : goodput {cub[0]:.2f}x | lat-infl {cub[1]:.2f}x | loss {cub[2]*100:.2f}%")
    print(f"  AIMD      : goodput {aim[0]:.2f}x | lat-infl {aim[1]:.2f}x | loss {aim[2]*100:.2f}%")

    # --- honest verdict ---
    print("\n" + "=" * 74)
    print("HOW TO READ THIS")
    print("=" * 74)
    print(f"* If goodput std is small (say < 0.05) and every seed's lat-infl is")
    print(f"  well below CUBIC's {cub[1]:.2f}x -> the win REPRODUCES. Real result.")
    print(f"* If goodput swings wildly across seeds -> the single good run was")
    print(f"  partly luck; the setup needs stabilizing before building further.")
    print(f"* Remember: this is ONE link, single-flow, vs CUBIC/AIMD (not BBR).")
    print(f"  A reproducing win here is a real START, not 'world-class' yet.")


if __name__ == "__main__":
    main()