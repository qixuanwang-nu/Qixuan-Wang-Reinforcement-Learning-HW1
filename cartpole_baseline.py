import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ==================== Policy Network (Actor) ====================

class CartPolePolicy(nn.Module):
    """Policy network (Actor) for CartPole environment"""
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super(CartPolePolicy, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, action_dim)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.softmax(x, dim=-1)


# ==================== Value Network (Critic/Baseline) ====================

class ValueNetwork(nn.Module):
    """Value network (Critic) to estimate V(s) as baseline"""
    def __init__(self, state_dim, hidden_dim=128):
        super(ValueNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x.squeeze(-1)  # Output scalar value


# ==================== Helper Functions ====================

def compute_returns(rewards, gamma):
    """Compute discounted returns for each timestep"""
    returns = []
    R = 0
    for r in reversed(rewards):
        R = r + gamma * R
        returns.insert(0, R)
    returns = torch.tensor(returns, dtype=torch.float32).to(device)
    return returns


def moving_average(data, window_size):
    """Compute moving average"""
    if len(data) < window_size:
        return np.array([np.mean(data[:i+1]) for i in range(len(data))])
    
    moving_avg = []
    for i in range(len(data)):
        if i < window_size:
            moving_avg.append(np.mean(data[:i+1]))
        else:
            moving_avg.append(np.mean(data[i-window_size+1:i+1]))
    return np.array(moving_avg)


# ==================== Policy Gradient with Baseline ====================

def train_policy_gradient_baseline(env_name, policy, value_net, policy_optimizer, 
                                   value_optimizer, gamma, num_episodes, max_steps=None):
    """
    Train policy using REINFORCE with learned value function baseline (Actor-Critic)
    
    The advantage is computed as: A_t = G_t - V(s_t)
    This reduces variance while keeping the gradient unbiased.
    """
    env = gym.make(env_name)
    episode_rewards = []
    
    for episode in range(num_episodes):
        state, _ = env.reset()
        
        log_probs = []
        values = []
        rewards = []
        states = []
        
        done = False
        step = 0
        
        while not done:
            # Convert state to tensor
            state_tensor = torch.FloatTensor(state).to(device)
            
            # Get action probabilities and value estimate
            action_probs = policy(state_tensor)
            value = value_net(state_tensor)
            
            # Sample action from the distribution
            dist = Categorical(action_probs)
            action = dist.sample()
            log_prob = dist.log_prob(action)
            
            # Take action in environment
            next_state, reward, terminated, truncated, _ = env.step(action.item())
            done = terminated or truncated
            
            # Store data
            log_probs.append(log_prob)
            values.append(value)
            rewards.append(reward)
            states.append(state_tensor)
            
            state = next_state
            step += 1
            
            if max_steps and step >= max_steps:
                break
        
        # Compute returns (targets for value function)
        returns = compute_returns(rewards, gamma)
        
        # Convert to tensors
        log_probs = torch.stack(log_probs)
        values = torch.stack(values)
        
        # Compute advantages: A_t = G_t - V(s_t)
        advantages = returns - values.detach()  # Detach to prevent gradient flow to value net
        
        # Normalize advantages for stability (optional but helps)
        if len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Policy loss: -log π(a|s) * A(s,a)
        policy_loss = -(log_probs * advantages).sum()
        
        # Value loss: MSE between V(s) and G_t
        value_loss = F.mse_loss(values, returns)
        
        # Optimize policy network (actor)
        policy_optimizer.zero_grad()
        policy_loss.backward()
        policy_optimizer.step()
        
        # Optimize value network (critic)
        value_optimizer.zero_grad()
        value_loss.backward()
        value_optimizer.step()
        
        # Record episode reward
        episode_reward = sum(rewards)
        episode_rewards.append(episode_reward)
        
        # Print progress
        if (episode + 1) % 100 == 0:
            avg_reward = np.mean(episode_rewards[-100:])
            avg_advantage = advantages.mean().item()
            print(f"Episode {episode + 1}/{num_episodes}, "
                  f"Avg Reward (last 100): {avg_reward:.2f}, "
                  f"Avg Advantage: {avg_advantage:.3f}")
    
    env.close()
    return episode_rewards


def evaluate_policy(env_name, policy, num_episodes=500):
    """Evaluate trained policy over multiple episodes (greedy)"""
    env = gym.make(env_name)
    eval_rewards = []
    
    for episode in range(num_episodes):
        state, _ = env.reset()
        
        episode_reward = 0
        done = False
        
        while not done:
            state_tensor = torch.FloatTensor(state).to(device)
            
            with torch.no_grad():
                action_probs = policy(state_tensor)
                action = torch.argmax(action_probs).item()
            
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            episode_reward += reward
            state = next_state
        
        eval_rewards.append(episode_reward)
        
        if (episode + 1) % 100 == 0:
            print(f"Evaluated {episode + 1}/{num_episodes} episodes")
    
    env.close()
    return eval_rewards


def plot_results(episode_rewards, eval_rewards, title, save_prefix):
    """Plot training curve and evaluation histogram"""
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot training curve
    ax1 = axes[0]
    episodes = np.arange(1, len(episode_rewards) + 1)
    ma = moving_average(episode_rewards, 100)
    
    ax1.plot(episodes, episode_rewards, alpha=0.3, label='Episode Reward')
    ax1.plot(episodes, ma, linewidth=2, label='Moving Average (100 episodes)')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Reward')
    ax1.set_title(f'{title} - Training Curve')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot evaluation histogram
    ax2 = axes[1]
    mean_reward = np.mean(eval_rewards)
    std_reward = np.std(eval_rewards)
    
    ax2.hist(eval_rewards, bins=30, edgecolor='black', alpha=0.7)
    ax2.axvline(mean_reward, color='red', linestyle='--', linewidth=2, 
                label=f'Mean: {mean_reward:.2f}')
    ax2.set_xlabel('Episode Reward')
    ax2.set_ylabel('Frequency')
    ax2.set_title(f'{title} - Evaluation Histogram (500 episodes)\n'
                  f'Mean: {mean_reward:.2f}, Std: {std_reward:.2f}')
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(f'{save_prefix}_results.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print(f"\n{title} Evaluation Results:")
    print(f"Mean Reward: {mean_reward:.2f}")
    print(f"Std Reward: {std_reward:.2f}")


# ==================== Main Training Script ====================

def train_cartpole():
    """Train CartPole-v1 with Value Function Baseline"""
    print("\n" + "="*70)
    print("Training CartPole-v1 with Value Function Baseline (Actor-Critic)")
    print("="*70 + "\n")
    
    print("BASELINE EXPLANATION:")
    print("-" * 70)
    print("Baseline Type: Learned Value Function V(s)")
    print("Implementation: Actor-Critic architecture")
    print()
    print("How it works:")
    print("  1. Critic Network learns V(s) to estimate expected return from state s")
    print("  2. Advantage: A(s,a) = G_t - V(s_t) measures how much better action")
    print("     a is compared to the average action in state s")
    print("  3. Policy gradient uses advantages: ∇J = E[∇log π(a|s) * A(s,a)]")
    print("  4. This reduces variance because we subtract state-dependent baseline")
    print("     while keeping the gradient unbiased (E[∇log π(a|s) * V(s)] = 0)")
    print()
    print("Benefits:")
    print("  - Reduces variance → more stable training")
    print("  - Faster convergence → learns more efficiently")
    print("  - Better final performance → advantage normalization helps")
    print("-" * 70 + "\n")
    
    # Environment parameters
    env = gym.make('CartPole-v1')
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    env.close()
    
    # Hyperparameters (same as Part 1 for fair comparison)
    gamma = 0.95
    learning_rate = 0.01
    num_episodes = 1000
    
    # Initialize networks
    policy = CartPolePolicy(state_dim, action_dim).to(device)
    value_net = ValueNetwork(state_dim).to(device)
    
    # Initialize optimizers
    policy_optimizer = optim.Adam(policy.parameters(), lr=learning_rate)
    value_optimizer = optim.Adam(value_net.parameters(), lr=learning_rate)
    
    print(f"Hyperparameters:")
    print(f"  Discount factor (γ): {gamma}")
    print(f"  Learning rate: {learning_rate}")
    print(f"  Episodes: {num_episodes}")
    print(f"  Policy network: {state_dim} → 128 → {action_dim}")
    print(f"  Value network: {state_dim} → 128 → 1")
    print()
    
    # Train
    episode_rewards = train_policy_gradient_baseline(
        'CartPole-v1', policy, value_net, policy_optimizer, 
        value_optimizer, gamma, num_episodes
    )
    
    # Evaluate
    print("\nEvaluating trained policy...")
    eval_rewards = evaluate_policy('CartPole-v1', policy, num_episodes=500)
    
    # Plot results
    plot_results(episode_rewards, eval_rewards, 
                'CartPole-v1 with Baseline', 'cartpole_baseline')
    
    # Save models
    torch.save({
        'policy_state_dict': policy.state_dict(),
        'value_state_dict': value_net.state_dict(),
    }, 'cartpole_baseline_policy.pth')
    print("\nModels saved as 'cartpole_baseline_policy.pth'")
    
    return policy, value_net, episode_rewards, eval_rewards


# ==================== Run Training ====================

if __name__ == "__main__":
    # Train CartPole with baseline
    policy, value_net, train_rewards, eval_rewards = train_cartpole()

