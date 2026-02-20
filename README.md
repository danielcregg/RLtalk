# Reinforcement Learning Talk

![Python](https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=flat-square&logo=jupyter&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=flat-square&logo=numpy&logoColor=white)

> **Note:** This repository is a fork of [thedataincubator/DS30_4](https://github.com/thedataincubator/DS30_4).

An interactive Jupyter Notebook presentation covering Reinforcement Learning fundamentals and Multi-Armed Bandit algorithms, originally prepared for The Data Incubator fellowship program.

## Overview

This notebook walks through core reinforcement learning concepts with live code examples, including epsilon-greedy strategies, Thompson sampling (Randomized Probability Matching), and a real-time Twitter streaming bandit experiment. The presentation covers theory from Sutton and Barto's textbook and connects it to practical applications such as ad optimization, debt collections, and content recommendation.

## Python 3 Standalone Scripts

The original notebook was written for Python 2.7 with deprecated dependencies (tweepy 3.5, etc.). These standalone Python 3 scripts extract and extend the core algorithms from the notebook into runnable form.

### `bandit_algorithms.py` - Multi-Armed Bandit Comparison

Implements and compares four bandit algorithms on a 10-armed Gaussian testbed:

| Algorithm | Source | Description |
|-----------|--------|-------------|
| Epsilon-Greedy | Original notebook | Explore randomly with probability epsilon |
| Thompson Sampling | Original notebook | Bayesian approach with Beta-Bernoulli posteriors |
| **UCB** | **New extension** | Upper Confidence Bound (Auer et al., 2002) |
| **Exp3** | **New extension** | Exponential-weight algorithm for adversarial bandits |

Generates three comparison plots: average reward, cumulative regret, and optimal action percentage.

```bash
python bandit_algorithms.py
```

### `mdp_examples.py` - MDP Theory Implementations

Provides working code for the MDP theory covered in the presentation's markdown-only cells (cells 29-32):

| Method | Description |
|--------|-------------|
| Value Iteration | Solves the Bellman Optimality Equation iteratively |
| Policy Iteration | Alternates between policy evaluation and improvement |
| TD(0) Prediction | Temporal-difference learning for value estimation |

All methods are demonstrated on a 4x4 grid world (Sutton & Barto, Example 4.1).

```bash
python mdp_examples.py
```

## Features

- **Multi-Armed Bandit Simulation** -- Implementation of a 10-armed bandit testbed with configurable epsilon-greedy exploration
- **Epsilon-Greedy Strategy** -- Comparison of different epsilon values (0.0, 0.01, 0.1) with averaged reward plots
- **Thompson Sampling** -- Bayesian approach using Beta distribution posteriors for action selection
- **UCB (New)** -- Upper Confidence Bound for optimistic exploration
- **Exp3 (New)** -- Adversarial bandit algorithm with theoretical regret guarantees
- **MDP Solvers (New)** -- Value Iteration, Policy Iteration, and TD(0) with convergence visualizations
- **Live Twitter Bandit** -- Real-time streaming experiment (original notebook, requires Python 2.7 and Twitter API)
- **Industry Case Studies** -- Examples from Yahoo! Labs article recommendation and NY State tax collection

## Prerequisites

- Python 3.8 or higher
- pip package manager

## Getting Started

```bash
# Install dependencies
pip install -r requirements.txt

# Run bandit comparison
python bandit_algorithms.py

# Run MDP examples
python mdp_examples.py

# Run tests
pytest tests/ -v
```

Plots are saved to the `images/` directory.

## Running Tests

```bash
# Run all tests
pytest tests/ -v

# Test bandit algorithms only
pytest tests/test_bandits.py -v

# Test MDP examples only
pytest tests/test_mdp.py -v
```

## Tech Stack

- **Python** -- Core language for all implementations
- **NumPy** -- Numerical computations and bandit simulations
- **Matplotlib** -- Visualization of reward curves and posterior distributions
- **SciPy** -- Beta distribution for Thompson sampling
- **Jupyter Notebook** -- Interactive presentation format (original notebook)

## License

This repository is a fork of [thedataincubator/DS30_4](https://github.com/thedataincubator/DS30_4). Please refer to the upstream repository for licensing information.
