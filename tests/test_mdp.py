#######################################################################
# Tests for MDP Examples
#
# Verifies that:
# 1. Value Iteration converges to correct values
# 2. Policy Iteration agrees with Value Iteration
# 3. TD(0) prediction converges
# 4. Grid world environment works correctly
#######################################################################

import sys
import os
import numpy as np
import pytest

# Add repo root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mdp_examples import (
    GridWorld, value_iteration, policy_iteration, policy_evaluation,
    td0_prediction, generate_episode, run_quick_test
)


class TestGridWorld:
    """Tests for the GridWorld environment."""

    def test_creation(self):
        env = GridWorld(size=4)
        assert env.n_states == 16
        assert env.n_actions == 4
        assert 0 in env.terminal_states
        assert 15 in env.terminal_states

    def test_terminal_state_stays(self):
        env = GridWorld(size=4)
        for action in range(4):
            next_s, reward = env.step(0, action)
            assert next_s == 0
            assert reward == 0.0

    def test_non_terminal_reward(self):
        env = GridWorld(size=4)
        _, reward = env.step(5, 0)
        assert reward == -1.0

    def test_boundary_stays(self):
        env = GridWorld(size=4)
        # State 1 (top row), going up should stay
        next_s, _ = env.step(1, 0)  # up
        assert next_s == 1

    def test_movement(self):
        env = GridWorld(size=4)
        # State 5 (row 1, col 1), going right should go to state 6
        next_s, _ = env.step(5, 3)  # right
        assert next_s == 6

    def test_transitions_sum_to_one(self):
        env = GridWorld(size=4)
        for s in range(env.n_states):
            for a in range(env.n_actions):
                transitions = env.get_transitions(s, a)
                total_prob = sum(p for p, _, _ in transitions)
                assert abs(total_prob - 1.0) < 1e-10


class TestValueIteration:
    """Tests for Value Iteration."""

    def test_converges(self):
        env = GridWorld(size=4)
        V, policy, history = value_iteration(env)
        assert len(history) > 0
        assert history[-1] < 1e-6  # converged

    def test_terminal_values_zero(self):
        env = GridWorld(size=4)
        V, _, _ = value_iteration(env)
        assert V[0] == 0.0
        assert V[15] == 0.0

    def test_non_terminal_values_negative(self):
        env = GridWorld(size=4)
        V, _, _ = value_iteration(env)
        for s in range(env.n_states):
            if s not in env.terminal_states:
                assert V[s] < 0

    def test_known_values(self):
        """Test against known values for 4x4 grid world."""
        env = GridWorld(size=4)
        V, _, _ = value_iteration(env)
        # State 1 (adjacent to terminal 0) should be approximately -1
        assert abs(V[1] - (-1.0)) < 0.1
        # State 5 (diagonal from terminal) should be approximately -2
        assert abs(V[5] - (-2.0)) < 0.1


class TestPolicyIteration:
    """Tests for Policy Iteration."""

    def test_converges(self):
        env = GridWorld(size=4)
        V, policy, n_iters = policy_iteration(env)
        assert n_iters > 0

    def test_agrees_with_value_iteration(self):
        env = GridWorld(size=4)
        V_vi, _, _ = value_iteration(env)
        V_pi, _, _ = policy_iteration(env)
        assert np.allclose(V_vi, V_pi, atol=1e-4)

    def test_optimal_policy_reaches_terminal(self):
        """Following the optimal policy from any state should reach terminal."""
        env = GridWorld(size=4)
        _, policy, _ = policy_iteration(env)
        for start_s in range(env.n_states):
            if start_s in env.terminal_states:
                continue
            s = start_s
            for _ in range(20):
                if s in env.terminal_states:
                    break
                s, _ = env.step(s, policy[s])
            assert s in env.terminal_states, (
                f"Policy did not reach terminal from state {start_s}"
            )


class TestTD0:
    """Tests for TD(0) Prediction."""

    def test_runs(self):
        env = GridWorld(size=4)
        _, policy, _ = value_iteration(env)
        V, rmse = td0_prediction(env, policy, n_episodes=50, alpha=0.1)
        assert len(rmse) == 50

    def test_rmse_decreases(self):
        """RMSE should generally decrease over many episodes."""
        env = GridWorld(size=4)
        _, policy, _ = value_iteration(env)
        V, rmse = td0_prediction(env, policy, n_episodes=500, alpha=0.1)
        early_rmse = np.mean(rmse[:50])
        late_rmse = np.mean(rmse[-50:])
        assert late_rmse < early_rmse

    def test_terminal_values_stay_zero(self):
        env = GridWorld(size=4)
        _, policy, _ = value_iteration(env)
        V, _ = td0_prediction(env, policy, n_episodes=100, alpha=0.1)
        assert V[0] == 0.0
        assert V[15] == 0.0


class TestGenerateEpisode:
    """Tests for episode generation."""

    def test_returns_list(self):
        env = GridWorld(size=4)
        _, policy, _ = value_iteration(env)
        episode = generate_episode(env, policy)
        assert isinstance(episode, list)
        assert len(episode) > 0

    def test_episode_format(self):
        env = GridWorld(size=4)
        _, policy, _ = value_iteration(env)
        episode = generate_episode(env, policy)
        for state, action, reward, next_state in episode:
            assert 0 <= state < env.n_states
            assert 0 <= action < env.n_actions
            assert reward == -1.0 or state in env.terminal_states


class TestQuickTest:
    """Test the built-in quick test function."""

    def test_quick_test(self):
        run_quick_test()
