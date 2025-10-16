import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np
import ale_py
import gymnasium as gym
import matplotlib.pyplot as plt
from collections import deque

# Register ALE environments
gym.register_envs(ale_py)

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ==================== Policy Networks ====================

class CartPolePolicy(nn.Module):
    """Policy network for CartPole environment"""
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super(CartPolePolicy, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, action_dim)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize network weights"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0.0)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.softmax(x, dim=-1)


class PongPolicy(nn.Module):
    """Policy network for Pong with CNN architecture"""
    def __init__(self, action_dim=2):
        super(PongPolicy, self).__init__()
        # CNN layers for processing 80x80 images
        self.conv1 = nn.Conv2d(1, 16, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=4, stride=2)
        
        # Calculate size after convolutions: 80 -> 19 -> 8
        self.fc1 = nn.Linear(32 * 8 * 8, 256)
        self.fc2 = nn.Linear(256, action_dim)
        
        # Initialize weights for better training stability
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize network weights with proper initialization"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0.0)
    
    def forward(self, x):
        # x shape: (batch, 80, 80) -> add channel dimension
        if len(x.shape) == 2:
            x = x.unsqueeze(0).unsqueeze(0)
        elif len(x.shape) == 3:
            x = x.unsqueeze(1)
        
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.softmax(x, dim=-1)


# ==================== Helper Functions ====================

def preprocess(image):
    """Prepro 210x160x3 uint8 frame into 6400 (80x80) 2D float array"""
    image = image[35:195]  # crop
    image = image[::2, ::2, 0]  # downsample by factor of 2
    image[image == 144] = 0  # erase background (background type 1)
    image[image == 109] = 0  # erase background (background type 2)
    image[image != 0] = 1  # everything else (paddles, ball) just set to 1
    return np.reshape(image.astype(float).ravel(), [80, 80])


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


# ==================== Policy Gradient Algorithm ====================

def train_policy_gradient(env_name, policy, optimizer, gamma, num_episodes, 
                         max_steps=None, is_pong=False, action_map=None):
    """
    Train policy using REINFORCE algorithm
    
    Args:
        env_name: Name of the gym environment
        policy: Policy network
        optimizer: PyTorch optimizer
        gamma: Discount factor
        num_episodes: Number of training episodes
        max_steps: Maximum steps per episode (None for default)
        is_pong: Whether this is Pong environment
        action_map: Mapping from policy action to env action (for Pong)
    """
    env = gym.make(env_name)
    episode_rewards = []
    
    for episode in range(num_episodes):
        state, _ = env.reset()
        
        # Preprocess state for Pong
        if is_pong:
            state = preprocess(state)
            prev_frame = None  # Track previous frame for motion
        
        log_probs = []
        rewards = []
        
        done = False
        step = 0
        
        while not done:
            # For Pong, use frame difference (motion signal)
            if is_pong:
                cur_frame = state
                if prev_frame is not None:
                    state_input = cur_frame - prev_frame
                else:
                    state_input = np.zeros_like(cur_frame, dtype=np.float32)
                prev_frame = cur_frame
                state_tensor = torch.FloatTensor(state_input).to(device)
            else:
                # Convert state to tensor
                state_tensor = torch.FloatTensor(state).to(device)
            
            # Get action probabilities
            action_probs = policy(state_tensor)
            
            # Sample action from the distribution
            dist = Categorical(action_probs)
            action = dist.sample()
            log_prob = dist.log_prob(action)
            
            # Map action for Pong (0,1 -> 2,3)
            if is_pong:
                env_action = action_map[action.item()]
            else:
                env_action = action.item()
            
            # Take action in environment
            next_state, reward, terminated, truncated, _ = env.step(env_action)
            done = terminated or truncated
            
            # Preprocess next state for Pong
            if is_pong:
                next_state = preprocess(next_state)
            
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
        # Gradient clipping for training stability
        torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=1.0)
        optimizer.step()
        
        # Record episode reward
        episode_reward = sum(rewards)
        episode_rewards.append(episode_reward)
        
        # Print progress
        if (episode + 1) % 100 == 0:
            avg_reward = np.mean(episode_rewards[-100:])
            print(f"Episode {episode + 1}/{num_episodes}, "
                  f"Avg Reward (last 100): {avg_reward:.2f}")
        
        # Save checkpoint for Pong every 500 episodes
        if is_pong and (episode + 1) % 500 == 0:
            checkpoint_path = f'pong_checkpoint_ep{episode + 1}.pth'
            torch.save({
                'episode': episode + 1,
                'policy_state_dict': policy.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'episode_rewards': episode_rewards,
            }, checkpoint_path)
            print(f"  → Checkpoint saved: {checkpoint_path}")
    
    env.close()
    return episode_rewards


def evaluate_policy(env_name, policy, num_episodes=500, is_pong=False, action_map=None):
    """Evaluate trained policy over multiple episodes"""
    env = gym.make(env_name)
    eval_rewards = []
    
    for episode in range(num_episodes):
        state, _ = env.reset()
        
        if is_pong:
            state = preprocess(state)
            prev_frame = None  # Track previous frame for motion
        
        episode_reward = 0
        done = False
        
        while not done:
            # For Pong, use frame difference (motion signal)
            if is_pong:
                cur_frame = state
                if prev_frame is not None:
                    state_input = cur_frame - prev_frame
                else:
                    state_input = np.zeros_like(cur_frame, dtype=np.float32)
                prev_frame = cur_frame
                state_tensor = torch.FloatTensor(state_input).to(device)
            else:
                state_tensor = torch.FloatTensor(state).to(device)
            
            with torch.no_grad():
                action_probs = policy(state_tensor)
                action = torch.argmax(action_probs).item()
            
            if is_pong:
                env_action = action_map[action]
            else:
                env_action = action
            
            next_state, reward, terminated, truncated, _ = env.step(env_action)
            done = terminated or truncated
            
            if is_pong:
                next_state = preprocess(next_state)
            
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


# ==================== Main Training Scripts ====================

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


def train_pong():
    """Train Pong-v5"""
    print("\n" + "="*60)
    print("Training Pong-v5")
    print("="*60 + "\n")
    
    # Hyperparameters
    gamma = 0.99
    learning_rate = 0.001  # Lower learning rate for stability
    num_episodes = 1000  # Pong requires more episodes
    
    # Action mapping: policy outputs 0 or 1, map to RIGHT(2) or LEFT(3)
    action_map = [2, 3]  # Index 0->RIGHT(2), Index 1->LEFT(3)
    
    # Initialize policy and optimizer
    policy = PongPolicy(action_dim=2).to(device)
    optimizer = optim.Adam(policy.parameters(), lr=learning_rate)
    
    print(f"Using learning rate: {learning_rate} (reduced for stability)")
    print(f"Action mapping: 0->RIGHT(2), 1->LEFT(3)")
    print(f"Gradient clipping: max_norm=1.0")
    print(f"Weight initialization: Kaiming (Conv) + Xavier (FC)\n")
    
    # Train with periodic checkpointing
    print("Starting training (checkpoints saved every 500 episodes)...\n")
    episode_rewards = train_policy_gradient(
        'ALE/Pong-v5', policy, optimizer, gamma, num_episodes,
        is_pong=True, action_map=action_map
    )
    
    print("\nTraining completed!")
    
    # Evaluate
    print("\nEvaluating trained policy...")
    eval_rewards = evaluate_policy(
        'ALE/Pong-v5', policy, num_episodes=500,
        is_pong=True, action_map=action_map
    )
    
    # Plot results
    plot_results(episode_rewards, eval_rewards, 'Pong-v5', 'pong')
    
    # Save model
    torch.save(policy.state_dict(), 'pong_policy.pth')
    print("\nModel saved as 'pong_policy.pth'")
    
    return policy, episode_rewards, eval_rewards


# ==================== Run Training ====================

if __name__ == "__main__":
    # Train CartPole
    #cartpole_policy, cartpole_train_rewards, cartpole_eval_rewards = train_cartpole()
    
    # Train Pong (this will take longer)
    #print("\n\nNote: Pong training will take significantly longer (may take hours)")
    #print("You may want to reduce num_episodes if just testing the code.\n")
    
    # Uncomment the line below to train Pong
    pong_policy, pong_train_rewards, pong_eval_rewards = train_pong()
