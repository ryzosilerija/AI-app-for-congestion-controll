"""
compare_fairness.py — did cross-traffic training improve CUBIC-sharing?

Loads both agents (original agent.pt and cross-trained agent_cross.pt) and runs
each against CUBIC on the shared bottleneck, side by side, so you can see if the
fix worked: PPO-vs-CUBIC Jain should rise from ~0.81 toward fair (1.0).

Run:  python compare_fairness.py
(Requires both agent.pt and agent_cross.pt to exist.)
"""

from __future__ import annotations
import os
import numpy as np

from fairness import run_two_flows, jain, PPOControllerWrapper, LINK_BPS
from train_and_save import load_agent
from controllers import Cubic


def test_vs_cubic(model, label):
    ppo = PPOControllerWrapper(model, LINK_BPS)
    ta, tb = run_two_flows(ppo, Cubic(), label, "CUBIC")
    return ta, tb, jain([ta, tb])


def main():
    if not (os.path.exists("agent.pt") and os.path.exists("agent_cross.pt")):
        print("Need both agent.pt and agent_cross.pt.")
        print("Run train_and_save.py and train_cross.py first.")
        return

    orig = load_agent("agent.pt")
    cross = load_agent("agent_cross.pt")

    print("=" * 60)
    print("FAIRNESS vs CUBIC  |  original vs cross-traffic-trained agent")
    print("=" * 60)
    print(f"{'agent':>22} | {'PPO':>8} {'CUBIC':>8} | {'Jain':>6}")
    print("-" * 60)

    for model, label in [(orig, "original (agent.pt)"),
                         (cross, "cross-trained")]:
        ta, tb, j = test_vs_cubic(model, "PPO")
        print(f"{label:>22} | {ta:6.2f}Mb {tb:6.2f}Mb | {j:5.3f}")

    print("-" * 60)
    print("If cross-trained Jain > original's 0.81, the fix worked: the agent")
    print("learned to hold its share against CUBIC. If PPO throughput rose and")
    print("CUBIC's fell toward equal, that's the diagnose->fix->validate loop.")


if __name__ == "__main__":
    main()
    