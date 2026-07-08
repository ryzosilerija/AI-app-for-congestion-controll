Learned Congestion Control

A reinforcement-learning congestion controller built from scratch in Python. The
controller is trained with PPO to manage a custom reliable transport protocol,
and is evaluated against the standard baselines (TCP Reno/AIMD, CUBIC, and BBR)
across a range of network conditions.

On a standard link it matches CUBIC's throughput while keeping queuing latency
close to the physical minimum, and it maintains low loss where BBR does not.

Motivation

Classical congestion control algorithms react to a single fixed signal. Loss-based
schemes (Reno, CUBIC) fill the bottleneck buffer until a packet drops, which
achieves high throughput at the cost of high queuing latency (bufferbloat).
Delay-based and model-based schemes (BBR) target the bandwidth-delay product to
keep latency low, but make their own tradeoffs. This project investigates whether
a controller can learn a policy from experience that achieves a better overall
throughput-latency-loss balance, and how well such a learned policy generalizes to
conditions it was not trained on.

What is implemented

The project is built in layers, each independently testable:


A discrete-event network emulator (channel.py) modelling propagation delay, a
finite bottleneck queue, and packet loss.
A reliable transport protocol over that channel (transport.py): sequence
numbers, cumulative ACKs, Jacobson/Karels RTT estimation, retransmission on
timeout and on triple-duplicate-ACK, and pacing.
A pluggable congestion-control interface (controllers.py) with baseline
implementations of Fixed-Window, AIMD/Reno, CUBIC, and BBR.
A reinforcement-learning environment wrapping the transport (rl_env.py),
exposing the network state and accepting a congestion-window action, with a
PCC-style utility reward (throughput penalized by latency inflation and loss).
A PPO agent (train_ppo.py) implemented in PyTorch (actor-critic, GAE, clipped
objective).
Evaluation harnesses for multi-seed validation, generalization to unseen
conditions, full baseline comparison, and multi-flow fairness.


Method

The state observed by the agent each control interval is: latency inflation
(smoothed RTT over minimum RTT), minimum RTT, normalized throughput, instantaneous
loss rate, current window, in-flight ratio, and a smoothed loss-history term. The
action is a multiplicative adjustment to the congestion window. The reward is:

reward = throughput_norm - LAMBDA * (latency_inflation - 1) - MU * loss_rate

The weights LAMBDA and MU encode the priority given to latency and loss relative
to throughput; they were tuned by a sweep (see Results).

Results

All figures below are single-flow on a 10 Mbps, 20 ms one-way link unless noted.
Latency is reported as inflation over the minimum RTT (1.00x = no queuing delay).

Baseline comparison

ControllerGoodputLatency inflationLossPPO (learned)0.95x1.03x0.0%CUBIC0.92x1.43x0.1%BBR0.91x1.00x48.0%AIMD/Reno0.87x1.07x0.1%

The learned controller matches or exceeds the baselines on throughput while
keeping both latency and loss low. CUBIC pays for its throughput in latency
(bufferbloat); BBR achieves low latency but drives very high loss because it does
not treat loss as a congestion signal.

Reproducibility

Trained across five independent random seeds, the learned controller converges
reliably: goodput 0.95x +/- 0.03, latency inflation 1.03x +/- 0.05, loss 0.0%.
Different initializations follow different learning paths but reach the same
operating point.

Generalization

The controller was trained on one link configuration and evaluated, without
retraining, on conditions it had never seen (different bandwidths, RTTs, buffer
sizes, and loss rates). It generalizes well across bandwidth, RTT, and buffer size,
and after training with domain randomization it beats CUBIC and BBR on most unseen
conditions. High-loss links (3% random loss) remain the hardest case and are the
main documented limitation.

Fairness (multi-flow)

Two flows sharing a 20 Mbps bottleneck, measured by Jain's fairness index
(1.0 = perfectly equal split):

MatchupJain indexPPO vs PPO1.00PPO vs BBR1.00PPO vs CUBIC0.81BBR vs CUBIC0.59

The controller is perfectly fair to copies of itself and to BBR. It under-shares
against CUBIC's aggressive buffer-filling (the classic delay-based-controller
vulnerability), but is notably more CUBIC-friendly than BBR is.

Development narrative

The project was developed as an iterative diagnose-fix-validate loop:


Built and validated the controller; confirmed reproducibility across seeds.
Tested generalization; found the controller memorized its training bandwidth
and had no loss-handling because it never saw loss in training.
Applied domain randomization (varied conditions per episode); the bandwidth and
RTT generalization gaps closed.
Found the loss penalty then caused over-conservative behavior; tuned the loss
weight via a sweep to recover throughput without reintroducing loss.
Characterized multi-flow fairness; identified and documented the CUBIC-sharing
limitation.


Limitations and future work


High-loss links (around 3%) remain difficult; the controller trades throughput
for loss control but does not fully resolve the regime.
The controller under-shares against CUBIC in multi-flow settings. A share-based
reward for competitive episodes is the intended next step.
All evaluation is in a custom discrete-event simulator. Porting the transport to
real UDP sockets and validating over emulated real networks (netem/Mahimahi,
Pantheon) is future work.


Running it

Requires Python 3.11+, numpy, and torch.

python simulate.py            # baseline controllers (AIMD, CUBIC) and the sawtooth
python train_ppo.py           # train the PPO controller (single run)
python validate_multiseed.py  # reproducibility across seeds
python train_and_save.py      # train one agent and save it to agent.pt
python compare_all.py         # PPO vs CUBIC vs BBR vs AIMD across conditions
python fairness.py            # multi-flow fairness (loads agent.pt)

Repository layout

channel.py              network emulator (delay, bottleneck queue, loss)
transport.py            reliable transport (ACKs, RTT estimation, retransmit)
controllers.py          controller interface + AIMD, CUBIC, BBR baselines
rl_env.py               RL environment, reward, domain randomization, cross-traffic
train_ppo.py            PPO agent (actor-critic, GAE, clipped objective)
train_and_save.py       train and persist an agent
validate_multiseed.py   multi-seed reproducibility
generalize.py           generalization to unseen conditions
compare_all.py          full baseline comparison
fairness.py             multi-flow fairness
telemetry.py            structured logging schema
