import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
from collections import deque

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ==================== Policy Network (CartPole) ====================

class CartPolePolicy(nn.Module):
    """Policy network for CartPole environment"""
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super(CartPolePolicy, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, action_dim)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.softmax(x, dim=-1)


# ==================== Helper Functions ====================

def compute_returns(rewards, gamma):
    """Compute discounted returns for each timestep"""
    returns = []
    R = 0
    for r in reversed(rewards):
        R = r + gamma * R
        returns.insert(0, R)
    returns = torch.tensor(returns, dtype=torch.float32).to(device)
    # Normalize returns for more stable training
    if len(returns) > 1:
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)
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


# ==================== Policy Gradient Algorithm (CartPole only) ====================

def train_policy_gradient(env_name, policy, optimizer, gamma, num_episodes, max_steps=None):
    """
    Train policy using REINFORCE algorithm (CartPole only)
    """
    env = gym.make(env_name)
    episode_rewards = []
    
    for episode in range(num_episodes):
        state, _ = env.reset()
        
        log_probs = []
        rewards = []
        
        done = False
        step = 0
        
        while not done:
            # Convert state to tensor
            state_tensor = torch.FloatTensor(state).to(device)
            
            # Get action probabilities
            action_probs = policy(state_tensor)
            
            # Sample action from the distribution
            dist = Categorical(action_probs)
            action = dist.sample()
            log_prob = dist.log_prob(action)
            
            # Take action in environment
            next_state, reward, terminated, truncated, _ = env.step(action.item())
            done = terminated or truncated
            
            # Store log probability and reward
            log_probs.append(log_prob)
            rewards.append(reward)
            
            state = next_state
            step += 1
            
            if max_steps and step >= max_steps:
                break
        
        # Compute returns
        returns = compute_returns(rewards, gamma)
        
        # Compute policy gradient loss
        policy_loss = []
        for log_prob, R in zip(log_probs, returns):
            policy_loss.append(-log_prob * R)
        
        # Optimize policy
        optimizer.zero_grad()
        policy_loss = torch.stack(policy_loss).sum()
        policy_loss.backward()
        optimizer.step()
        
        # Record episode reward
        episode_reward = sum(rewards)
        episode_rewards.append(episode_reward)
        
        # Print progress
        if (episode + 1) % 100 == 0:
            avg_reward = np.mean(episode_rewards[-100:])
            print(f"Episode {episode + 1}/{num_episodes}, "
                  f"Avg Reward (last 100): {avg_reward:.2f}")
    
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


# ==================== Main Training Script (CartPole only) ====================

def train_cartpole():
    """Train CartPole-v1"""
    print("\n" + "="*60)
    print("Training CartPole-v1")
    print("="*60 + "\n")
    
    # Environment parameters
    env = gym.make('CartPole-v1')
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    env.close()
    
    # Hyperparameters
    gamma = 0.95
    learning_rate = 0.01
    num_episodes = 1000
    
    # Initialize policy and optimizer
    policy = CartPolePolicy(state_dim, action_dim).to(device)
    optimizer = optim.Adam(policy.parameters(), lr=learning_rate)
    
    # Train
    episode_rewards = train_policy_gradient(
        'CartPole-v1', policy, optimizer, gamma, num_episodes
    )
    
    # Evaluate
    print("\nEvaluating trained policy...")
    eval_rewards = evaluate_policy('CartPole-v1', policy, num_episodes=500)
    
    # Plot results
    plot_results(episode_rewards, eval_rewards, 'CartPole-v1', 'cartpole')
    
    # Save model
    torch.save(policy.state_dict(), 'cartpole_policy.pth')
    print("\nModel saved as 'cartpole_policy.pth'")
    
    return policy, episode_rewards, eval_rewards


# ==================== Run Training ====================

if __name__ == "__main__":
    # Train CartPole only
    cartpole_policy, cartpole_train_rewards, cartpole_eval_rewards = train_cartpole()
