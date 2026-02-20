#######################################################################
# Bandit Algorithms: Python 3 Implementations of Multi-Armed Bandits
#
# This script provides Python 3 implementations of the bandit algorithms
# from the RLtalk presentation, plus two research extensions:
#
# Original (from notebook):
#   1. Epsilon-Greedy
#   2. Thompson Sampling (Beta-Bernoulli / RPM)
#
# New extensions:
#   3. UCB (Upper Confidence Bound)
#   4. Exp3 (Exponential-weight algorithm for Exploration and Exploitation)
#
# All four algorithms are compared on a standard 10-armed testbed
# with Gaussian rewards, producing average reward curves and cumulative
# regret plots.
#######################################################################

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# 10-Armed Testbed
# ---------------------------------------------------------------------------
class BanditTestbed:
    """A k-armed bandit testbed with Gaussian rewards.

    Each arm has a true mean reward drawn from N(0, 1). When pulled,
    an arm returns its true mean plus N(0, 1) noise.
    """
    def __init__(self, k=10):
        self.k = k
        self.arm_values = np.random.normal(0, 1, k)
        self.optimal_action = np.argmax(self.arm_values)
        self.optimal_value = self.arm_values[self.optimal_action]

    def pull(self, action):
        """Pull arm `action` and return a noisy reward."""
        return self.arm_values[action] + np.random.normal(0, 1)


# ---------------------------------------------------------------------------
# Algorithm 1: Epsilon-Greedy (from the original notebook)
# ---------------------------------------------------------------------------
class EpsilonGreedy:
    """Epsilon-greedy bandit algorithm.

    With probability epsilon, explore (random arm); otherwise exploit
    (arm with highest estimated value). Uses incremental sample-average
    updates, matching the notebook's implementation.
    """
    def __init__(self, k=10, epsilon=0.1):
        self.k = k
        self.epsilon = epsilon
        self.counts = np.zeros(k)
        self.values = np.zeros(k)

    def select_action(self):
        if np.random.random() < self.epsilon:
            return np.random.randint(self.k)
        else:
            return np.argmax(self.values)

    def update(self, action, reward):
        self.counts[action] += 1
        alpha = 1.0 / self.counts[action]
        self.values[action] += alpha * (reward - self.values[action])

    @property
    def name(self):
        return f"Epsilon-Greedy (eps={self.epsilon})"


# ---------------------------------------------------------------------------
# Algorithm 2: Thompson Sampling (Beta-Bernoulli)
# ---------------------------------------------------------------------------
class ThompsonSampling:
    """Thompson Sampling with Beta-Bernoulli posteriors.

    Based on the Randomized Probability Matching (RPM) from the notebook.
    Rewards are converted to binary (positive = success) to fit the
    Beta-Bernoulli model. Each arm maintains Beta(alpha, beta) posterior
    parameters.
    """
    def __init__(self, k=10, prior_alpha=1.0, prior_beta=1.0):
        self.k = k
        self.alphas = np.full(k, prior_alpha)
        self.betas = np.full(k, prior_beta)

    def select_action(self):
        samples = np.random.beta(self.alphas, self.betas)
        return np.argmax(samples)

    def update(self, action, reward):
        # Convert continuous reward to binary: positive = success
        if reward > 0:
            self.alphas[action] += 1
        else:
            self.betas[action] += 1

    @property
    def name(self):
        return "Thompson Sampling"


# ---------------------------------------------------------------------------
# Algorithm 3: UCB (Upper Confidence Bound) - NEW EXTENSION
# ---------------------------------------------------------------------------
class UCB:
    """UCB1 (Upper Confidence Bound) algorithm.

    Selects the arm that maximizes: Q(a) + c * sqrt(ln(t) / N(a))
    where Q(a) is the estimated value, t is the total number of pulls,
    N(a) is the number of times arm a has been pulled, and c controls
    the exploration bonus.

    Reference: Auer et al., "Finite-time Analysis of the Multiarmed
    Bandit Problem", Machine Learning, 2002.
    """
    def __init__(self, k=10, c=2.0):
        self.k = k
        self.c = c
        self.counts = np.zeros(k)
        self.values = np.zeros(k)
        self.total_pulls = 0

    def select_action(self):
        self.total_pulls += 1
        # Pull each arm once first
        for a in range(self.k):
            if self.counts[a] == 0:
                return a
        # UCB selection
        ucb_values = self.values + self.c * np.sqrt(
            np.log(self.total_pulls) / self.counts
        )
        return np.argmax(ucb_values)

    def update(self, action, reward):
        self.counts[action] += 1
        alpha = 1.0 / self.counts[action]
        self.values[action] += alpha * (reward - self.values[action])

    @property
    def name(self):
        return f"UCB (c={self.c})"


# ---------------------------------------------------------------------------
# Algorithm 4: Exp3 (Exponential-weight for Exploration and Exploitation)
#              - NEW EXTENSION
# ---------------------------------------------------------------------------
class Exp3:
    """Exp3 algorithm for adversarial multi-armed bandits.

    Unlike the other algorithms which assume stochastic rewards, Exp3
    provides guarantees even when rewards are chosen adversarially.
    It maintains a probability distribution over arms using exponential
    weights and importance-weighted reward estimates.

    Reference: Auer et al., "The Nonstochastic Multiarmed Bandit Problem",
    SIAM Journal on Computing, 2002.
    """
    def __init__(self, k=10, gamma=0.1):
        self.k = k
        self.gamma = gamma  # mixing parameter for exploration
        self.weights = np.ones(k)

    def _get_probs(self):
        total = np.sum(self.weights)
        probs = (1 - self.gamma) * (self.weights / total) + self.gamma / self.k
        return probs

    def select_action(self):
        probs = self._get_probs()
        return np.random.choice(self.k, p=probs)

    def update(self, action, reward):
        probs = self._get_probs()
        # Normalize reward to [0, 1] range (assuming rewards roughly in [-3, 3])
        normalized_reward = (reward + 3) / 6.0
        normalized_reward = np.clip(normalized_reward, 0, 1)
        # Importance-weighted reward estimate
        estimated_reward = normalized_reward / probs[action]
        self.weights[action] *= np.exp(self.gamma * estimated_reward / self.k)
        # Prevent numerical overflow
        self.weights /= np.max(self.weights)

    @property
    def name(self):
        return f"Exp3 (gamma={self.gamma})"


# ---------------------------------------------------------------------------
# Experiment Runner
# ---------------------------------------------------------------------------
def run_experiment(algorithm_class, algorithm_kwargs, n_arms=10,
                   n_pulls=3000, n_experiments=500):
    """Run a bandit algorithm across multiple experiments.

    Returns:
        avg_rewards: Average reward at each timestep across experiments.
        avg_regret: Average cumulative regret at each timestep.
        avg_optimal: Fraction of times optimal action was chosen.
    """
    avg_rewards = np.zeros(n_pulls)
    avg_regret = np.zeros(n_pulls)
    avg_optimal = np.zeros(n_pulls)

    for exp in range(n_experiments):
        testbed = BanditTestbed(k=n_arms)
        algo = algorithm_class(k=n_arms, **algorithm_kwargs)
        cumulative_regret = 0.0

        for t in range(n_pulls):
            action = algo.select_action()
            reward = testbed.pull(action)
            algo.update(action, reward)

            avg_rewards[t] += reward
            cumulative_regret += (testbed.optimal_value - reward)
            avg_regret[t] += cumulative_regret
            avg_optimal[t] += (1.0 if action == testbed.optimal_action else 0.0)

    avg_rewards /= n_experiments
    avg_regret /= n_experiments
    avg_optimal /= n_experiments

    return avg_rewards, avg_regret, avg_optimal


def run_all_comparisons(n_arms=10, n_pulls=3000, n_experiments=500,
                        save_dir=None):
    """Run all four algorithms and generate comparison plots."""
    if save_dir is None:
        save_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), 'images'
        )
    os.makedirs(save_dir, exist_ok=True)

    # Define algorithms to compare
    algorithms = [
        (EpsilonGreedy, {'epsilon': 0.0}, 'Greedy (eps=0)'),
        (EpsilonGreedy, {'epsilon': 0.01}, 'Eps-Greedy (eps=0.01)'),
        (EpsilonGreedy, {'epsilon': 0.1}, 'Eps-Greedy (eps=0.1)'),
        (ThompsonSampling, {}, 'Thompson Sampling'),
        (UCB, {'c': 2.0}, 'UCB (c=2)'),
        (Exp3, {'gamma': 0.1}, 'Exp3 (gamma=0.1)'),
    ]

    results = {}
    for algo_cls, kwargs, label in algorithms:
        print(f"Running {label}...")
        rewards, regret, optimal = run_experiment(
            algo_cls, kwargs,
            n_arms=n_arms, n_pulls=n_pulls, n_experiments=n_experiments
        )
        results[label] = (rewards, regret, optimal)

    # --- Plot 1: Average Reward Curves ---
    plt.figure(figsize=(12, 6))
    colors = ['gray', 'blue', 'green', 'red', 'purple', 'orange']
    for (label, (rewards, _, _)), color in zip(results.items(), colors):
        plt.plot(rewards, label=label, alpha=0.8, color=color)
    plt.xlabel('Steps')
    plt.ylabel('Average Reward')
    plt.title('10-Armed Testbed: Average Reward Comparison')
    plt.ylim(0, 2.2)
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(save_dir, 'bandit_average_rewards.png')
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path}")

    # --- Plot 2: Cumulative Regret Curves ---
    plt.figure(figsize=(12, 6))
    for (label, (_, regret, _)), color in zip(results.items(), colors):
        plt.plot(regret, label=label, alpha=0.8, color=color)
    plt.xlabel('Steps')
    plt.ylabel('Cumulative Regret')
    plt.title('10-Armed Testbed: Cumulative Regret Comparison')
    plt.legend(loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(save_dir, 'bandit_cumulative_regret.png')
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path}")

    # --- Plot 3: Optimal Action Percentage ---
    plt.figure(figsize=(12, 6))
    for (label, (_, _, optimal)), color in zip(results.items(), colors):
        plt.plot(optimal * 100, label=label, alpha=0.8, color=color)
    plt.xlabel('Steps')
    plt.ylabel('% Optimal Action')
    plt.title('10-Armed Testbed: Optimal Action Selection Rate')
    plt.ylim(0, 100)
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(save_dir, 'bandit_optimal_action.png')
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path}")

    return results


def run_quick_test(n_pulls=50, n_experiments=5):
    """Quick test to verify all algorithms run without errors."""
    algorithms = [
        (EpsilonGreedy, {'epsilon': 0.1}),
        (ThompsonSampling, {}),
        (UCB, {'c': 2.0}),
        (Exp3, {'gamma': 0.1}),
    ]
    for algo_cls, kwargs in algorithms:
        rewards, regret, optimal = run_experiment(
            algo_cls, kwargs,
            n_arms=10, n_pulls=n_pulls, n_experiments=n_experiments
        )
        assert len(rewards) == n_pulls
        assert len(regret) == n_pulls
        assert len(optimal) == n_pulls
        assert regret[-1] >= 0  # regret should be non-negative
    print("Quick test passed: all bandit algorithms work correctly.")


if __name__ == '__main__':
    run_all_comparisons(n_pulls=3000, n_experiments=500)
