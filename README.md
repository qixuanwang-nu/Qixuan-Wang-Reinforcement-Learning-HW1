# Reinforcement Learning Homework 1: Policy Gradient Methods

This repository contains implementations of policy gradient algorithms (REINFORCE) for CartPole-v1 and Pong-v5 environments, with and without baseline methods for variance reduction.

## Overview

We implement the REINFORCE algorithm and demonstrate variance reduction using baseline techniques:
- **CartPole-v1**: Actor-Critic baseline (learned value function)
- **Pong-v5**: Moving Average baseline (exponential moving average of returns)

## Environment Setup

Install dependencies:
```bash
pip install -r requirements.txt
```

## Implementations

### Part 1: REINFORCE without Baseline

#### CartPole-v1 (`cartpole.py`)
- **Algorithm**: REINFORCE with return normalization
- **Network**: 2-layer MLP (4 → 128 → 2)
- **Hyperparameters**: γ=0.95, lr=0.01, 1000 episodes
- **Results**: Mean reward 492.96, Std 16.55

#### Pong-v5 (`pong.py`)
- **Algorithm**: REINFORCE with return normalization
- **Network**: CNN architecture for 80×80 preprocessed frames
- **Hyperparameters**: γ=0.99, lr=0.001, 1000 episodes
- **Results**: Mean reward 3.27, Std 4.19

### Part 2: REINFORCE with Baseline

#### CartPole-v1 with Actor-Critic Baseline (`cartpole_baseline.py`)

**Baseline Type**: Learned state-value function V(s)

**Implementation**:
- Separate value network (critic) estimates V(s)
- Advantage computed as A(s,a) = G_t - V(s_t)
- Policy gradient uses advantages: ∇J = E[∇log π(a|s) · A(s,a)]
- Both actor and critic trained with separate Adam optimizers (lr=0.01)

**Architecture**:
- Actor: State(4) → FC(128) → FC(2) → Softmax
- Critic: State(4) → FC(128) → FC(1) → V(s)

**Results**: Mean reward 138.76, Std 8.18
- Variance reduced by 50% (16.55 → 8.18)
- Performance degradation due to hyperparameter sensitivity

#### Pong-v5 with Moving Average Baseline (`pong_baseline.py`)

**Baseline Type**: Exponential moving average of episode returns

**Implementation**:
- Maintains running average: b_t = β·b_{t-1} + (1-β)·G_episode
- Advantage computed as A_t = G_t - b_t
- Simple scalar baseline shared across all timesteps
- No additional networks required

**Architecture**:
- Policy network: CNN (80×80 → Conv → Conv → FC → 2 actions)
- Baseline: Single scalar updated with EMA (β=0.95)

**Hyperparameters**: γ=0.99, lr=0.0003, entropy bonus=0.01

**Results**: Mean reward 4.55, Std 2.11
- Variance reduced by 75% in training, 50% in evaluation
- Performance improved by 39% (3.27 → 4.55)

## Results Visualization

### CartPole-v1 Results

**Original REINFORCE** (`cartpole_results.png`):
- Training curve shows high variance with episodes reaching maximum reward (500)
- Evaluation: Mean 492.96, Std 16.55
- Successfully solves the environment

**With Actor-Critic Baseline** (`cartpole_baseline_results.png`):
- Smoother training curve with reduced variance
- Evaluation: Mean 138.76, Std 8.18
- Variance reduction achieved but performance degraded due to critic instability

### Pong-v5 Results

**Original REINFORCE** (`pong_results.png`):
- Training curve gradually improves from -20 to ~3 over 1000 episodes
- High variance throughout training
- Evaluation: Mean 3.27, Std 4.19

**With Moving Average Baseline** (`pong_baseline_comparison.png`):
- Significantly smoother training curve
- Faster convergence and more stable learning
- Evaluation: Mean 4.55, Std 2.11
- Demonstrates effective variance reduction while maintaining performance

## Key Findings

### Baseline Comparison

| Environment | Baseline Type | Variance Reduction | Performance Change |
|-------------|---------------|-------------------|-------------------|
| CartPole | Actor-Critic (V(s)) | ✓ 50% std reduction | ⚠️ Degraded (hyperparameters) |
| Pong | Moving Average | ✓ 50% std reduction | ✓ +39% improvement |

### Why Different Baselines?

**CartPole - Actor-Critic**:
- State-dependent baseline provides maximum variance reduction
- Low-dimensional state space (4D) makes critic easy to learn
- Demonstrates the Actor-Critic framework

**Pong - Moving Average**:
- Simple and stable for high-dimensional visual input
- No additional network to train → more stable
- Effective variance reduction without hyperparameter sensitivity
- Computationally efficient

## Mathematical Foundation

### Standard REINFORCE:
```
∇J = E[Σ_t ∇log π(a_t|s_t) · G_t]
```

### REINFORCE with Baseline:
```
∇J = E[Σ_t ∇log π(a_t|s_t) · (G_t - b_t)]
```

The baseline b_t reduces variance while keeping the gradient unbiased:
```
E[∇log π(a|s) · b] = 0
```

## Running the Code

### Train CartPole (Original):
```bash
python cartpole.py
```

### Train CartPole with Baseline:
```bash
python cartpole_baseline.py
```

### Train Pong (Original):
```bash
python pong.py
```

### Train Pong with Baseline:
```bash
python pong_baseline.py
```

## Files

### Code:
- `cartpole.py` - CartPole with REINFORCE
- `cartpole_baseline.py` - CartPole with Actor-Critic baseline
- `pong.py` - Pong with REINFORCE
- `pong_baseline.py` - Pong with Moving Average baseline

### Results:
- `cartpole_results.png` - CartPole training and evaluation plots
- `cartpole_baseline_results.png` - CartPole with baseline plots
- `pong_results.png` - Pong training and evaluation plots
- `pong_baseline_comparison.png` - Pong with baseline comparison plots

### Model:
- `pong_policy.pth` - Trained Pong policy weights

### Dependencies:
- `requirements.txt` - Python package requirements

## Key Takeaways

1. **Baseline reduces variance**: Both methods show significant reduction in gradient variance
2. **Performance trade-offs**: Moving average baseline is more stable than learned baselines
3. **Hyperparameter sensitivity**: Actor-Critic requires careful tuning
4. **Environment-specific choices**: Different baselines work better for different environments
5. **Variance-bias trade-off**: Baseline choice affects both training stability and final performance

## References

- Sutton & Barto, "Reinforcement Learning: An Introduction"
- Williams (1992), "Simple Statistical Gradient-Following Algorithms for Connectionist Reinforcement Learning"
- Mnih et al. (2016), "Asynchronous Methods for Deep Reinforcement Learning"

