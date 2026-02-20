#######################################################################
# Tests for Bandit Algorithms
#
# Verifies that each bandit algorithm:
# 1. Runs without errors
# 2. Produces reasonable results (non-trivial rewards)
# 3. Has correct API (select_action, update methods)
#######################################################################

import sys
import os
import numpy as np
import pytest

# Add repo root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bandit_algorithms import (
    BanditTestbed, EpsilonGreedy, ThompsonSampling, UCB, Exp3,
    run_experiment, run_quick_test
)


class TestBanditTestbed:
    """Tests for the bandit testbed environment."""

    def test_creation(self):
        testbed = BanditTestbed(k=10)
        assert len(testbed.arm_values) == 10
        assert 0 <= testbed.optimal_action < 10

    def test_pull_returns_float(self):
        testbed = BanditTestbed(k=10)
        reward = testbed.pull(0)
        assert isinstance(reward, (int, float, np.floating))

    def test_optimal_action_is_best(self):
        testbed = BanditTestbed(k=10)
        assert testbed.arm_values[testbed.optimal_action] == max(testbed.arm_values)


class TestEpsilonGreedy:
    """Tests for the epsilon-greedy algorithm."""

    def test_greedy_action_selection(self):
        algo = EpsilonGreedy(k=5, epsilon=0.0)
        algo.values = np.array([0.1, 0.5, 0.3, 0.2, 0.4])
        action = algo.select_action()
        assert action == 1  # highest value

    def test_exploration(self):
        """With epsilon=1.0, all actions should be selected eventually."""
        algo = EpsilonGreedy(k=5, epsilon=1.0)
        actions_seen = set()
        for _ in range(200):
            actions_seen.add(algo.select_action())
        assert len(actions_seen) == 5

    def test_update(self):
        algo = EpsilonGreedy(k=5, epsilon=0.1)
        algo.update(0, 1.0)
        assert algo.counts[0] == 1
        assert algo.values[0] == 1.0
        algo.update(0, 3.0)
        assert algo.counts[0] == 2
        assert algo.values[0] == 2.0  # running average

    def test_experiment_produces_results(self):
        rewards, regret, optimal = run_experiment(
            EpsilonGreedy, {'epsilon': 0.1},
            n_arms=10, n_pulls=100, n_experiments=5
        )
        assert len(rewards) == 100
        assert regret[-1] >= 0


class TestThompsonSampling:
    """Tests for Thompson Sampling."""

    def test_select_action_in_range(self):
        algo = ThompsonSampling(k=10)
        for _ in range(50):
            action = algo.select_action()
            assert 0 <= action < 10

    def test_update_positive(self):
        algo = ThompsonSampling(k=5)
        algo.update(2, 1.0)  # positive reward
        assert algo.alphas[2] == 2.0
        assert algo.betas[2] == 1.0

    def test_update_negative(self):
        algo = ThompsonSampling(k=5)
        algo.update(2, -1.0)  # negative reward
        assert algo.alphas[2] == 1.0
        assert algo.betas[2] == 2.0

    def test_experiment_produces_results(self):
        rewards, regret, optimal = run_experiment(
            ThompsonSampling, {},
            n_arms=10, n_pulls=100, n_experiments=5
        )
        assert len(rewards) == 100


class TestUCB:
    """Tests for UCB algorithm."""

    def test_initial_exploration(self):
        """UCB should try each arm once before exploiting."""
        algo = UCB(k=5, c=2.0)
        first_actions = []
        for _ in range(5):
            a = algo.select_action()
            algo.update(a, 0.0)
            first_actions.append(a)
        assert set(first_actions) == {0, 1, 2, 3, 4}

    def test_ucb_bonus_decreases(self):
        """The UCB bonus should decrease as an arm is pulled more."""
        algo = UCB(k=5, c=2.0)
        # Pull all arms once
        for a in range(5):
            algo.select_action()
            algo.update(a, 0.0)
        # Pull arm 0 many more times
        for _ in range(100):
            algo.total_pulls += 1
            algo.update(0, 0.0)
        # UCB bonus for arm 0 should be small, others should be large
        bonus_0 = algo.c * np.sqrt(np.log(algo.total_pulls) / algo.counts[0])
        bonus_1 = algo.c * np.sqrt(np.log(algo.total_pulls) / algo.counts[1])
        assert bonus_0 < bonus_1

    def test_experiment_produces_results(self):
        rewards, regret, optimal = run_experiment(
            UCB, {'c': 2.0},
            n_arms=10, n_pulls=100, n_experiments=5
        )
        assert len(rewards) == 100
        assert regret[-1] >= 0


class TestExp3:
    """Tests for Exp3 algorithm."""

    def test_select_action_in_range(self):
        algo = Exp3(k=10, gamma=0.1)
        for _ in range(50):
            action = algo.select_action()
            assert 0 <= action < 10

    def test_probabilities_sum_to_one(self):
        algo = Exp3(k=5, gamma=0.1)
        probs = algo._get_probs()
        assert abs(sum(probs) - 1.0) < 1e-10

    def test_initial_uniform(self):
        algo = Exp3(k=5, gamma=0.1)
        probs = algo._get_probs()
        assert np.allclose(probs, 0.2, atol=1e-10)

    def test_experiment_produces_results(self):
        rewards, regret, optimal = run_experiment(
            Exp3, {'gamma': 0.1},
            n_arms=10, n_pulls=100, n_experiments=5
        )
        assert len(rewards) == 100


class TestQuickTest:
    """Test the built-in quick test function."""

    def test_quick_test(self):
        run_quick_test(n_pulls=20, n_experiments=3)
