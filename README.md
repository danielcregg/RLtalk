# Reinforcement Learning Talk

![Python](https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=flat-square&logo=jupyter&logoColor=white)

> **Note:** This repository is a fork of [thedataincubator/DS30_4](https://github.com/thedataincubator/DS30_4).

An interactive Jupyter Notebook presentation covering Reinforcement Learning fundamentals and Multi-Armed Bandit algorithms, originally prepared for The Data Incubator fellowship program.

## Overview

This notebook walks through core reinforcement learning concepts with live code examples, including epsilon-greedy strategies, Thompson sampling (Randomized Probability Matching), and a real-time Twitter streaming bandit experiment. The presentation covers theory from Sutton and Barto's textbook and connects it to practical applications such as ad optimization, debt collections, and content recommendation.

## Features

- **Multi-Armed Bandit Simulation** — Implementation of a 10-armed bandit testbed with configurable epsilon-greedy exploration
- **Epsilon-Greedy Strategy** — Comparison of different epsilon values (0.0, 0.01, 0.1) with averaged reward plots
- **Thompson Sampling** — Bayesian approach using Beta distribution posteriors for action selection
- **Live Twitter Bandit** — Real-time streaming experiment that applies bandit algorithms to incoming tweets
- **MDP Theory** — Coverage of Markov Decision Processes, dynamic programming, Monte Carlo methods, and TD learning
- **Industry Case Studies** — Examples from Yahoo! Labs article recommendation and NY State tax collection optimization

## Prerequisites

- Python 2.7 (as specified in the Conda environment)
- [Anaconda](https://www.anaconda.com/products/distribution) or Miniconda
- Twitter API developer credentials (for the streaming example)

## Getting Started

### Installation

1. Create the Conda environment from the provided configuration:

   ```bash
   conda env create -f environment.yml
   ```

2. Activate the environment:

   ```bash
   source activate RLtalk
   ```

3. (Optional) For the Twitter streaming example, create a `keys.txt` file with your API credentials (one per line):
   - Consumer Key
   - Consumer Secret
   - Access Token
   - Access Token Secret

   Sign up at [Twitter Developer](https://developer.twitter.com/) to obtain credentials.

### Usage

Launch Jupyter Notebook and open the presentation:

```bash
jupyter notebook RLtalk.ipynb
```

## Tech Stack

- **Python** — Core language for all implementations
- **NumPy** — Numerical computations and bandit simulations
- **Matplotlib** — Visualization of reward curves and posterior distributions
- **SciPy** — Beta distribution for Thompson sampling
- **Tweepy** — Twitter API streaming integration
- **Jupyter Notebook** — Interactive presentation format

## License

This repository is a fork of [thedataincubator/DS30_4](https://github.com/thedataincubator/DS30_4). Please refer to the upstream repository for licensing information.
