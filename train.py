"""
train.py — the RL agent + training loop (items 16-17).

This is a NumPy-only policy-gradient agent (REINFORCE with a moving-average
baseline and a Gaussian policy). Why NumPy instead of a deep PPO?
  * Zero install pain — runs anywhere, no torch/gym version conflicts.
  * It genuinely LEARNS, proving the whole loop (env + reward + updates) works.
  * It's a clean drop-in target: when you want a deeper agent, replace the
    `LinearGaussianPolicy` with a PyTorch network + PPO clip objective. The
    env, reward, and training-loop scaffolding stay identical.

What to expect: the agent starts near-random (worse than AIMD), then the
average episode return should climb as it learns to track the bottleneck while
keeping the queue (latency) low. Beating AIMD decisively is the multi-week
research phase — this proves the machinery is sound.
"""

from __future__ import annotations
import numpy as np
from rl_env import TransportEnv


class LinearGaussianPolicy:
    """pi(a|s) = Normal(mean = W·s + b, std). Linear features, learned by
    policy gradient. The math: grad_log_prob * advantage, averaged over a batch.
    This is exactly the policy-gradient theorem, just with a linear mean."""

    def __init__(self, obs_dim, act_dim=1, lr=0.01, init_std=0.5, seed=0):
        rng = np.random.default_rng(seed)
        self.W = rng.normal(0, 0.1, size=(act_dim, obs_dim))
        self.b = np.zeros(act_dim)
        self.log_std = np.log(init_std) * np.ones(act_dim)
        self.lr = lr

    def mean(self, obs):
        return self.W @ obs + self.b

    def act(self, obs, explore=True):
        mu = self.mean(obs)
        std = np.exp(self.log_std)
        if explore:
            a = mu + std * np.random.standard_normal(mu.shape)
        else:
            a = mu
        return a, mu, std

    def grad_log_prob(self, obs, action):
        mu = self.mean(obs)
        std = np.exp(self.log_std)
        diff = (action - mu) / (std ** 2)
        # gradients wrt W, b, log_std
        gW = np.outer(diff, obs)
        gb = diff
        glog_std = (((action - mu) ** 2) / (std ** 2)) - 1.0
        return gW, gb, glog_std

    def update(self, batch, advantages):
        gW = np.zeros_like(self.W)
        gb = np.zeros_like(self.b)
        gls = np.zeros_like(self.log_std)
        n = len(batch)
        for (obs, action), adv in zip(batch, advantages):
            dW, db, dls = self.grad_log_prob(obs, action)
            gW += dW * adv
            gb += db * adv
            gls += dls * adv
        # gradient ASCENT on expected return
        self.W += self.lr * gW / n
        self.b += self.lr * gb / n
        self.log_std += self.lr * 0.1 * gls / n
        self.log_std = np.clip(self.log_std, np.log(0.05), np.log(1.0))


def run_episode(env, policy, explore=True, seed=None):
    obs = env.reset(seed=seed)
    traj, rewards, infos = [], [], []
    done = False
    while not done:
        a, mu, std = policy.act(obs, explore=explore)
        next_obs, r, done, info = env.step(a)
        traj.append((obs, a))
        rewards.append(r)
        infos.append(info)
        obs = next_obs
    return traj, rewards, infos


def train(episodes=300, gamma=0.99, log_every=20):
    env = TransportEnv(loss_rate=0.0, episode_s=8.0)
    policy = LinearGaussianPolicy(env.obs_dim, env.act_dim, lr=0.02, seed=0)
    baseline = 0.0
    history = []

    for ep in range(1, episodes + 1):
        traj, rewards, infos = run_episode(env, policy, explore=True, seed=ep)

        # discounted returns
        returns = np.zeros(len(rewards))
        G = 0.0
        for t in reversed(range(len(rewards))):
            G = rewards[t] + gamma * G
            returns[t] = G

        baseline = 0.9 * baseline + 0.1 * returns.mean()
        advantages = returns - baseline
        if advantages.std() > 1e-6:
            advantages = advantages / (advantages.std() + 1e-8)

        policy.update(traj, advantages)

        ep_return = sum(rewards)
        avg_goodput = np.mean([i["goodput_norm"] for i in infos])
        avg_infl = np.mean([i["inflation"] for i in infos])
        history.append((ep, ep_return, avg_goodput, avg_infl))

        if ep % log_every == 0 or ep == 1:
            print(f"ep {ep:4d} | return {ep_return:7.2f} | "
                  f"goodput {avg_goodput:4.2f}x link | "
                  f"latency-infl {avg_infl:4.2f}x | "
                  f"std {np.exp(policy.log_std)[0]:.2f}")

    return policy, history


def evaluate(policy, seed=999):
    env = TransportEnv(loss_rate=0.0, episode_s=8.0)
    _, rewards, infos = run_episode(env, policy, explore=False, seed=seed)
    gp = np.mean([i["goodput_norm"] for i in infos])
    infl = np.mean([i["inflation"] for i in infos])
    loss = np.mean([i["loss"] for i in infos])
    print("\n--- greedy evaluation of trained policy ---")
    print(f"  goodput        : {gp:.2f}x link capacity")
    print(f"  latency infl.  : {infl:.2f}x base RTT")
    print(f"  loss           : {loss*100:.2f}%")
    print(f"  return         : {sum(rewards):.2f}")


if __name__ == "__main__":
    print("Training NumPy policy-gradient congestion controller...")
    print("(starts near-random; watch return & goodput climb)\n")
    policy, history = train(episodes=300)
    evaluate(policy)
    print("\nNote: this linear agent proves the training loop works. For the")
    print("thesis-grade controller, swap LinearGaussianPolicy for a PyTorch")
    print("network with PPO — env/reward/loop stay the same.")