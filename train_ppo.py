"""
train_ppo.py — PyTorch PPO congestion controller (drop-in upgrade of train.py).

This replaces the NumPy LinearGaussianPolicy with a proper actor-critic PPO.
NOTHING about the environment, reward, or protocol changes — only the agent and
the update rule. That's the payoff of the clean controller interface.

WHAT'S NEW vs the linear REINFORCE agent, and WHY:

  1. ACTOR network  — a small MLP computes the action mean (was: one matrix).
     More expressive -> can learn nonlinear rules ("back off hard only when
     loss AND high RTT") a linear policy structurally cannot represent.

  2. CRITIC network — estimates V(s), "expected return from this state."
     The linear agent used a crude moving-average baseline; a learned critic
     gives a far less noisy advantage signal. THIS is the main reason PPO's
     returns stop bouncing the way the linear agent's did.

  3. GAE (Generalized Advantage Estimation) — smooths the advantage using the
     critic's value estimates and TD errors. The `gae_lambda` knob trades bias
     vs variance.

  4. PPO CLIPPED OBJECTIVE — limits how far the policy moves per update, so a
     single bad batch can't collapse the policy. Collect a batch, then do
     several minibatch gradient steps over it with the clip.

NOTE: requires `pip install torch` (you have it). Run with `python train_ppo.py`.
"""

from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn
from rl_env import TransportEnv


# --------------------------------------------------------------------------
# Actor-Critic networks
# --------------------------------------------------------------------------
class ActorCritic(nn.Module):
    """Separate small MLPs for policy (actor) and value (critic).

    The actor outputs the MEAN of a Gaussian over actions; log_std is a learned
    state-independent parameter (standard, stable choice for continuous PPO).
    The critic outputs a single scalar value estimate V(s)."""

    def __init__(self, obs_dim, act_dim=1, hidden=64):
        super().__init__()
        self.actor = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, act_dim),
        )
        self.critic = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, 1),
        )
        # log_std starts at exp(-0.5) ~= 0.6 exploration; learned during training.
        self.log_std = nn.Parameter(torch.full((act_dim,), -0.5))

    def get_dist(self, obs):
        mean = self.actor(obs)
        # Safety net: clamp to keep the distribution valid even if a bad value
        # slips through (NaN/inf mean or extreme log_std crashes Normal()).
        mean = torch.nan_to_num(mean, nan=0.0, posinf=10.0, neginf=-10.0)
        log_std = torch.clamp(self.log_std, -5.0, 2.0)
        std = torch.exp(log_std)
        return torch.distributions.Normal(mean, std)

    def value(self, obs):
        return self.critic(obs).squeeze(-1)


# --------------------------------------------------------------------------
# Rollout collection
# --------------------------------------------------------------------------
def collect_rollout(env, model, n_steps, device, seed=None):
    """Run the env, collecting (obs, action, logprob, reward, value, done).

    PPO is on-policy: we gather a batch with the CURRENT policy, update, then
    discard and gather fresh. We collect across episode boundaries up to
    n_steps transitions so each update sees a decent batch."""
    obs_buf, act_buf, logp_buf, rew_buf, val_buf, done_buf = [], [], [], [], [], []
    infos_collected = []

    obs = env.reset(seed=seed)
    for _ in range(n_steps):
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device)
        with torch.no_grad():
            dist = model.get_dist(obs_t)
            action = dist.sample()
            logp = dist.log_prob(action).sum(-1)
            value = model.value(obs_t)

        a_np = action.cpu().numpy()
        next_obs, reward, done, info = env.step(a_np)

        obs_buf.append(obs)
        act_buf.append(a_np)
        logp_buf.append(logp.item())
        rew_buf.append(reward)
        val_buf.append(value.item())
        done_buf.append(done)
        infos_collected.append(info)

        obs = next_obs
        if done:
            obs = env.reset(seed=(None if seed is None else seed + 1))

    # bootstrap value for the final state (for GAE on the truncated tail)
    obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device)
    with torch.no_grad():
        last_val = model.value(obs_t).item()

    return {
        "obs": np.array(obs_buf, dtype=np.float32),
        "act": np.array(act_buf, dtype=np.float32),
        "logp": np.array(logp_buf, dtype=np.float32),
        "rew": np.array(rew_buf, dtype=np.float32),
        "val": np.array(val_buf, dtype=np.float32),
        "done": np.array(done_buf, dtype=np.float32),
        "last_val": last_val,
        "infos": infos_collected,
    }


# --------------------------------------------------------------------------
# Generalized Advantage Estimation
# --------------------------------------------------------------------------
def compute_gae(rew, val, done, last_val, gamma=0.99, lam=0.95):
    """GAE: A_t = sum_l (gamma*lam)^l * delta_{t+l},
       delta_t = r_t + gamma*V(s_{t+1})*(1-done) - V(s_t).
    Returns advantages and value targets (advantages + values)."""
    n = len(rew)
    adv = np.zeros(n, dtype=np.float32)
    last_gae = 0.0
    for t in reversed(range(n)):
        next_val = last_val if t == n - 1 else val[t + 1]
        next_nonterminal = 1.0 - done[t]
        delta = rew[t] + gamma * next_val * next_nonterminal - val[t]
        last_gae = delta + gamma * lam * next_nonterminal * last_gae
        adv[t] = last_gae
    returns = adv + val
    return adv, returns


# --------------------------------------------------------------------------
# PPO update
# --------------------------------------------------------------------------
def ppo_update(model, optimizer, batch, adv, returns, device,
               clip=0.2, epochs=10, minibatch=64, ent_coef=0.0, vf_coef=0.5):
    obs = torch.as_tensor(batch["obs"], device=device)
    act = torch.as_tensor(batch["act"], device=device)
    old_logp = torch.as_tensor(batch["logp"], device=device)
    adv_t = torch.as_tensor(adv, device=device)
    ret_t = torch.as_tensor(returns, device=device)

    # normalize advantages (standard PPO trick, reduces variance)
    adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)

    n = len(obs)
    idx = np.arange(n)
    for _ in range(epochs):
        np.random.shuffle(idx)
        for start in range(0, n, minibatch):
            mb = idx[start:start + minibatch]
            mb_obs, mb_act = obs[mb], act[mb]
            mb_oldlogp, mb_adv, mb_ret = old_logp[mb], adv_t[mb], ret_t[mb]

            dist = model.get_dist(mb_obs)
            logp = dist.log_prob(mb_act).sum(-1)
            entropy = dist.entropy().sum(-1).mean()

            # ratio = pi_new / pi_old  (exp of log-prob difference)
            ratio = torch.exp(logp - mb_oldlogp)
            # the CLIPPED objective — the heart of PPO
            unclipped = ratio * mb_adv
            clipped = torch.clamp(ratio, 1 - clip, 1 + clip) * mb_adv
            policy_loss = -torch.min(unclipped, clipped).mean()

            value = model.value(mb_obs)
            value_loss = ((value - mb_ret) ** 2).mean()

            loss = policy_loss + vf_coef * value_loss - ent_coef * entropy

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 0.5)  # gradient clipping
            optimizer.step()


# --------------------------------------------------------------------------
# Training driver
# --------------------------------------------------------------------------
def train(total_updates=200, steps_per_update=2048, lr=3e-4,
          gamma=0.99, gae_lambda=0.95, log_every=10, device="cpu",
          env=None):
    if env is None:
        env = TransportEnv(loss_rate=0.0, episode_s=8.0)
    model = ActorCritic(env.obs_dim, env.act_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    history = []

    for upd in range(1, total_updates + 1):
        batch = collect_rollout(env, model, steps_per_update, device, seed=upd)
        adv, returns = compute_gae(batch["rew"], batch["val"], batch["done"],
                                   batch["last_val"], gamma, gae_lambda)
        ppo_update(model, optimizer, batch, adv, returns, device)

        infos = batch["infos"]
        avg_goodput = np.mean([i["goodput_norm"] for i in infos])
        avg_infl = np.mean([i["inflation"] for i in infos])
        avg_loss = np.mean([i["loss"] for i in infos])
        mean_rew = batch["rew"].mean()
        history.append((upd, mean_rew, avg_goodput, avg_infl))

        if upd % log_every == 0 or upd == 1:
            print(f"upd {upd:4d} | mean_rew {mean_rew:6.3f} | "
                  f"goodput {avg_goodput:4.2f}x link | "
                  f"lat-infl {avg_infl:4.2f}x | loss {avg_loss*100:4.1f}% | "
                  f"std {torch.exp(model.log_std).item():.2f}")

    return model, history


def evaluate(model, seed=999, device="cpu"):
    env = TransportEnv(loss_rate=0.0, episode_s=8.0)
    obs = env.reset(seed=seed)
    infos, rewards = [], []
    done = False
    while not done:
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device)
        with torch.no_grad():
            # greedy = use the distribution mean (no sampling)
            action = model.actor(obs_t).cpu().numpy()
        obs, r, done, info = env.step(action)
        rewards.append(r)
        infos.append(info)
    gp = np.mean([i["goodput_norm"] for i in infos])
    infl = np.mean([i["inflation"] for i in infos])
    loss = np.mean([i["loss"] for i in infos])
    print("\n--- greedy evaluation of trained PPO policy ---")
    print(f"  goodput        : {gp:.2f}x link capacity   (AIMD baseline ~0.87x)")
    print(f"  latency infl.  : {infl:.2f}x base RTT")
    print(f"  loss           : {loss*100:.2f}%")
    print(f"  mean reward    : {np.mean(rewards):.3f}")


if __name__ == "__main__":
    torch.manual_seed(0)
    np.random.seed(0)
    print("Training PyTorch PPO congestion controller...")
    print("(on-policy; watch mean_rew and goodput climb, lat-infl stay low)\n")
    model, history = train(total_updates=200)
    evaluate(model)
    print("\nIf goodput is still timid (<0.8x), the reward weights in rl_env.py")
    print("(LAMBDA, MU) are likely over-penalizing latency/loss. Tune those FIRST,")
    print("before reaching for bigger networks or more training.")