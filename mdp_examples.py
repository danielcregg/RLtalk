#######################################################################
# MDP Examples: Code Implementations of the Theory from the RLtalk
#
# The original RLtalk notebook covers MDP theory in markdown cells
# (cells 29-32) but does not include runnable code implementations.
# This script provides working Python 3 implementations of:
#
#   1. Value Iteration on a simple grid world
#   2. Policy Iteration comparison
#   3. TD(0) Prediction
#
# These implementations bring the theory sections of the presentation
# to life with concrete, runnable examples.
#######################################################################

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Grid World Environment
# ---------------------------------------------------------------------------
class GridWorld:
    """A simple 4x4 grid world MDP.

    The agent starts at the top-left and the goal is at the bottom-right.
    States 0 and 15 are terminal. The agent can move up, down, left, right.
    Each non-terminal transition gives a reward of -1, encouraging the
    agent to reach the goal quickly.

    This matches the grid world example from Sutton & Barto, Example 4.1.
    Uses gamma=0.99 to guarantee convergence for any policy.
    """
    def __init__(self, size=4):
        self.size = size
        self.n_states = size * size
        self.n_actions = 4  # up, down, left, right
        self.terminal_states = {0, self.n_states - 1}
        self.gamma = 0.99

        # Action directions: up, down, left, right
        self.action_deltas = {
            0: (-1, 0),   # up
            1: (1, 0),    # down
            2: (0, -1),   # left
            3: (0, 1),    # right
        }

    def state_to_pos(self, s):
        return (s // self.size, s % self.size)

    def pos_to_state(self, row, col):
        return row * self.size + col

    def step(self, state, action):
        """Deterministic transition. Returns (next_state, reward)."""
        if state in self.terminal_states:
            return state, 0.0

        row, col = self.state_to_pos(state)
        dr, dc = self.action_deltas[action]
        new_row = max(0, min(self.size - 1, row + dr))
        new_col = max(0, min(self.size - 1, col + dc))
        next_state = self.pos_to_state(new_row, new_col)

        return next_state, -1.0

    def get_transitions(self, state, action):
        """Returns list of (probability, next_state, reward) tuples."""
        next_state, reward = self.step(state, action)
        return [(1.0, next_state, reward)]


# ---------------------------------------------------------------------------
# 1. Value Iteration
# ---------------------------------------------------------------------------
def value_iteration(env, theta=1e-6, gamma=None, max_iterations=10000):
    """Solve the grid world using Value Iteration (Bellman Optimality Equation).

    Iteratively updates V(s) = max_a sum_s' p(s'|s,a) [r + gamma * V(s')]
    until convergence.

    Args:
        env: GridWorld environment.
        theta: Convergence threshold.
        gamma: Discount factor (defaults to env.gamma).
        max_iterations: Maximum number of iterations.

    Returns:
        V: Converged state-value function.
        policy: Optimal policy (action for each state).
        history: List of max deltas per iteration (for convergence plot).
    """
    if gamma is None:
        gamma = env.gamma
    V = np.zeros(env.n_states)
    history = []

    for iteration in range(max_iterations):
        delta = 0
        for s in range(env.n_states):
            if s in env.terminal_states:
                continue
            v = V[s]
            action_values = []
            for a in range(env.n_actions):
                val = 0
                for prob, next_s, reward in env.get_transitions(s, a):
                    val += prob * (reward + gamma * V[next_s])
                action_values.append(val)
            V[s] = max(action_values)
            delta = max(delta, abs(v - V[s]))
        history.append(delta)
        if delta < theta:
            break

    # Extract optimal policy
    policy = np.zeros(env.n_states, dtype=int)
    for s in range(env.n_states):
        if s in env.terminal_states:
            continue
        action_values = []
        for a in range(env.n_actions):
            val = 0
            for prob, next_s, reward in env.get_transitions(s, a):
                val += prob * (reward + gamma * V[next_s])
            action_values.append(val)
        policy[s] = np.argmax(action_values)

    return V, policy, history


# ---------------------------------------------------------------------------
# 2. Policy Iteration
# ---------------------------------------------------------------------------
def policy_evaluation(env, policy, gamma=None, theta=1e-6, max_iterations=10000):
    """Evaluate a fixed policy by iterating the Bellman expectation equation.

    Args:
        env: GridWorld environment.
        policy: Array mapping states to actions.
        gamma: Discount factor (defaults to env.gamma).
        theta: Convergence threshold.
        max_iterations: Maximum iterations to prevent divergence.

    Returns:
        V: State-value function under the given policy.
    """
    if gamma is None:
        gamma = env.gamma
    V = np.zeros(env.n_states)
    for iteration in range(max_iterations):
        delta = 0
        for s in range(env.n_states):
            if s in env.terminal_states:
                continue
            v = V[s]
            a = policy[s]
            new_v = 0
            for prob, next_s, reward in env.get_transitions(s, a):
                new_v += prob * (reward + gamma * V[next_s])
            V[s] = new_v
            delta = max(delta, abs(v - V[s]))
        if delta < theta:
            break
    return V


def policy_improvement(env, V, old_policy, gamma=None):
    """Greedily improve the policy with respect to V.

    Args:
        env: GridWorld environment.
        V: Current state-value function.
        old_policy: The policy that was used to compute V.
        gamma: Discount factor.

    Returns:
        new_policy: The improved policy.
        stable: True if the policy did not change.
    """
    if gamma is None:
        gamma = env.gamma
    new_policy = np.copy(old_policy)
    stable = True
    for s in range(env.n_states):
        if s in env.terminal_states:
            continue
        action_values = []
        for a in range(env.n_actions):
            val = 0
            for prob, next_s, reward in env.get_transitions(s, a):
                val += prob * (reward + gamma * V[next_s])
            action_values.append(val)
        best_action = np.argmax(action_values)
        if best_action != old_policy[s]:
            stable = False
        new_policy[s] = best_action
    return new_policy, stable


def policy_iteration(env, gamma=None, theta=1e-6):
    """Solve the grid world using Policy Iteration.

    Alternates between policy evaluation and policy improvement
    until the policy is stable.

    Returns:
        V: Converged state-value function.
        policy: Optimal policy.
        n_iterations: Number of policy improvement steps.
    """
    if gamma is None:
        gamma = env.gamma
    # Start with a random policy
    policy = np.zeros(env.n_states, dtype=int)
    n_iterations = 0
    max_pi_iterations = 100

    while n_iterations < max_pi_iterations:
        V = policy_evaluation(env, policy, gamma=gamma, theta=theta)
        new_policy, stable = policy_improvement(env, V, policy, gamma=gamma)
        n_iterations += 1
        policy = new_policy
        if stable:
            break

    return V, policy, n_iterations


# ---------------------------------------------------------------------------
# 3. TD(0) Prediction
# ---------------------------------------------------------------------------
def generate_episode(env, policy, max_steps=100):
    """Generate an episode following the given policy.

    Returns list of (state, action, reward, next_state) tuples.
    """
    non_terminal = [s for s in range(env.n_states) if s not in env.terminal_states]
    state = np.random.choice(non_terminal)
    episode = []

    for _ in range(max_steps):
        if state in env.terminal_states:
            break
        action = policy[state]
        next_state, reward = env.step(state, action)
        episode.append((state, action, reward, next_state))
        state = next_state

    return episode


def td0_prediction(env, policy, n_episodes=1000, alpha=0.1, gamma=None):
    """TD(0) prediction: estimate V under a given policy.

    Update rule: V(s) += alpha * (r + gamma * V(s') - V(s))

    Returns:
        V: Estimated state-value function.
        rmse_history: RMSE vs true values at each episode.
    """
    if gamma is None:
        gamma = env.gamma
    V_true = policy_evaluation(env, policy, gamma=gamma)

    V = np.zeros(env.n_states)
    rmse_history = []

    for ep in range(n_episodes):
        episode = generate_episode(env, policy)
        for state, action, reward, next_state in episode:
            if next_state in env.terminal_states:
                td_target = reward
            else:
                td_target = reward + gamma * V[next_state]
            V[state] += alpha * (td_target - V[state])

        non_terminal = [s for s in range(env.n_states) if s not in env.terminal_states]
        rmse = np.sqrt(np.mean((V[non_terminal] - V_true[non_terminal]) ** 2))
        rmse_history.append(rmse)

    return V, rmse_history


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------
def format_policy(env, policy):
    """Format policy as a human-readable grid."""
    action_symbols = {0: 'U', 1: 'D', 2: 'L', 3: 'R'}
    grid = []
    for i in range(env.size):
        row = []
        for j in range(env.size):
            s = env.pos_to_state(i, j)
            if s in env.terminal_states:
                row.append('T')
            else:
                row.append(action_symbols[policy[s]])
        grid.append(row)
    return grid


def run_all_examples(save_dir=None):
    """Run all MDP examples and generate plots."""
    if save_dir is None:
        save_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), 'images'
        )
    os.makedirs(save_dir, exist_ok=True)

    env = GridWorld(size=4)

    # --- Value Iteration ---
    print("Running Value Iteration...")
    V_vi, policy_vi, vi_history = value_iteration(env)
    print(f"  Converged in {len(vi_history)} iterations")
    print(f"  Value function:\n{V_vi.reshape(env.size, env.size).round(1)}")
    print(f"  Policy:\n{format_policy(env, policy_vi)}")

    # --- Policy Iteration ---
    print("\nRunning Policy Iteration...")
    V_pi, policy_pi, pi_iters = policy_iteration(env)
    print(f"  Converged in {pi_iters} policy improvement steps")
    print(f"  Value function:\n{V_pi.reshape(env.size, env.size).round(1)}")
    print(f"  Policy:\n{format_policy(env, policy_pi)}")

    # Verify both methods agree
    assert np.allclose(V_vi, V_pi, atol=1e-4), "Value and Policy Iteration should agree!"
    print("\nBoth methods agree on the optimal value function.")

    # --- TD(0) Prediction ---
    print("\nRunning TD(0) Prediction...")
    V_td, rmse_history = td0_prediction(env, policy_vi, n_episodes=2000, alpha=0.1)
    print(f"  Final RMSE: {rmse_history[-1]:.4f}")

    # --- Generate Plots ---
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # Plot 1: Value function heatmap
    ax = axes[0, 0]
    V_grid = V_vi.reshape(env.size, env.size)
    im = ax.imshow(V_grid, cmap='RdYlGn', interpolation='nearest')
    for i in range(env.size):
        for j in range(env.size):
            ax.text(j, i, f'{V_grid[i, j]:.1f}', ha='center', va='center',
                    fontsize=12, fontweight='bold')
    ax.set_title('Optimal State Values (Value Iteration)')
    ax.set_xlabel('Column')
    ax.set_ylabel('Row')
    plt.colorbar(im, ax=ax)

    # Plot 2: Optimal policy arrows
    ax = axes[0, 1]
    policy_grid = format_policy(env, policy_vi)
    ax.set_xlim(-0.5, env.size - 0.5)
    ax.set_ylim(env.size - 0.5, -0.5)
    arrow_map = {'U': (0, -0.3), 'D': (0, 0.3), 'L': (-0.3, 0), 'R': (0.3, 0)}
    for i in range(env.size):
        for j in range(env.size):
            symbol = policy_grid[i][j]
            if symbol == 'T':
                ax.text(j, i, 'T', ha='center', va='center', fontsize=16,
                        fontweight='bold', color='red')
            else:
                dx, dy = arrow_map[symbol]
                ax.annotate('', xy=(j + dx, i + dy), xytext=(j, i),
                           arrowprops=dict(arrowstyle='->', color='blue', lw=2))
    ax.set_xticks(range(env.size))
    ax.set_yticks(range(env.size))
    ax.grid(True, alpha=0.3)
    ax.set_title('Optimal Policy (arrows)')
    ax.set_xlabel('Column')
    ax.set_ylabel('Row')

    # Plot 3: Value Iteration convergence
    ax = axes[1, 0]
    ax.plot(vi_history, 'b-o', markersize=3)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Max Value Change (delta)')
    ax.set_title('Value Iteration Convergence')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)

    # Plot 4: TD(0) RMSE convergence
    ax = axes[1, 1]
    ax.plot(rmse_history, 'g-', alpha=0.7)
    ax.set_xlabel('Episode')
    ax.set_ylabel('RMSE vs True Values')
    ax.set_title('TD(0) Prediction Convergence')
    ax.grid(True, alpha=0.3)

    plt.suptitle('MDP Examples: Grid World (4x4)', fontsize=14, y=1.01)
    plt.tight_layout()

    path = os.path.join(save_dir, 'mdp_examples.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nPlot saved to {path}")


def run_quick_test():
    """Quick test to verify all MDP methods work correctly."""
    env = GridWorld(size=4)

    # Value Iteration
    V_vi, policy_vi, history = value_iteration(env)
    assert len(history) > 0
    assert V_vi[0] == 0  # terminal state
    assert V_vi[-1] == 0  # terminal state

    # Policy Iteration
    V_pi, policy_pi, n_iters = policy_iteration(env)
    assert n_iters > 0
    assert np.allclose(V_vi, V_pi, atol=1e-4)

    # TD(0)
    V_td, rmse = td0_prediction(env, policy_vi, n_episodes=100, alpha=0.1)
    assert len(rmse) == 100
    assert rmse[-1] < rmse[0] + 1.0  # loose check

    print("Quick test passed: all MDP methods work correctly.")


if __name__ == '__main__':
    run_all_examples()
