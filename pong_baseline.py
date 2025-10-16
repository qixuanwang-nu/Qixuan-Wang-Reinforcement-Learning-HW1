import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np
import ale_py
import gymnasium as gym
import matplotlib.pyplot as plt

# Register ALE environments
gym.register_envs(ale_py)

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ==================== Policy Network (Actor) ====================

class PongPolicy(nn.Module):
    """Policy network (Actor) for Pong with CNN architecture"""
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


# ==================== Value Network (Critic) ====================

class PongValueNetwork(nn.Module):
    """Value network (Critic) for Pong with CNN architecture"""
    def __init__(self):
        super(PongValueNetwork, self).__init__()
        # CNN layers (same as policy network)
        self.conv1 = nn.Conv2d(1, 16, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=4, stride=2)
        
        # Value head
        self.fc1 = nn.Linear(32 * 8 * 8, 256)
        self.fc2 = nn.Linear(256, 1)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize network weights"""
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
        return x.squeeze(-1)  # Output scalar value


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
    # Normalize returns to stabilize optimization (matches non-baseline script)
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


# ==================== Policy Gradient with Baseline ====================

def train_policy_gradient_baseline(env_name, policy, value_net, policy_optimizer,
                                   value_optimizer, gamma, num_episodes, action_map,
                                   entropy_coeff=0.01):
    """
    Train Pong using REINFORCE with learned value function baseline (Actor-Critic)
    
    The advantage is computed as: A_t = G_t - V(s_t)
    This reduces variance while keeping the gradient unbiased.
    """
    env = gym.make(env_name)
    episode_rewards = []
    
    for episode in range(num_episodes):
        state, _ = env.reset()
        state = preprocess(state)
        prev_frame = None  # Track previous frame for motion
        entropies = []
        
        log_probs = []
        values = []
        rewards = []
        
        done = False
        step = 0
        
        while not done:
            # Use frame difference (motion signal)
            cur_frame = state
            if prev_frame is not None:
                state_input = cur_frame - prev_frame
            else:
                state_input = np.zeros_like(cur_frame, dtype=np.float32)
            prev_frame = cur_frame
            
            state_tensor = torch.FloatTensor(state_input).to(device)
            
            # Get action probabilities and value estimate
            action_probs = policy(state_tensor)
            value = value_net(state_tensor)
            
            # Sample action from the distribution
            dist = Categorical(action_probs)
            action = dist.sample()
            log_prob = dist.log_prob(action)
            entropy = dist.entropy()
            
            # Map action for Pong (0,1 -> 2,3)
            env_action = action_map[action.item()]
            
            # Take action in environment
            next_state, reward, terminated, truncated, _ = env.step(env_action)
            done = terminated or truncated
            
            # Preprocess next state
            next_state = preprocess(next_state)
            
            # Store data
            log_probs.append(log_prob)
            entropies.append(entropy)
            values.append(value)
            rewards.append(reward)
            
            state = next_state
            step += 1
        
        # Compute returns (targets for value function)
        returns = compute_returns(rewards, gamma)
        
        # Convert to tensors
        log_probs = torch.stack(log_probs)
        entropies = torch.stack(entropies)
        values = torch.stack(values)
        
        # Ensure tensors have correct 1D shape [T]
        if len(values.shape) > 1:
            values = values.squeeze()
        if len(log_probs.shape) > 1:
            log_probs = log_probs.squeeze()
        
        # Compute advantages: A_t = G_t - V(s_t)
        advantages = returns - values.detach()  # Detach to prevent gradient flow to value net
        
        # Normalize advantages for stability
        if len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Policy loss with entropy bonus: -log π(a|s) * A(s,a) - β * H[π(·|s)]
        policy_loss = (-(log_probs * advantages) - entropy_coeff * entropies).sum()
        
        # Value loss: Smooth L1 (Huber) between V(s) and normalized G_t
        value_loss = F.smooth_l1_loss(values, returns)
        
        # Optimize policy network (actor)
        policy_optimizer.zero_grad()
        policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=1.0)
        policy_optimizer.step()
        
        # Optimize value network (critic)
        value_optimizer.zero_grad()
        value_loss.backward()
        torch.nn.utils.clip_grad_norm_(value_net.parameters(), max_norm=1.0)
        value_optimizer.step()
        
        # Record episode reward
        episode_reward = sum(rewards)
        episode_rewards.append(episode_reward)
        
        # Print progress
        if (episode + 1) % 100 == 0:
            avg_reward = np.mean(episode_rewards[-100:])
            avg_value_loss = value_loss.item()
            print(f"Episode {episode + 1}/{num_episodes}, "
                  f"Avg Reward (last 100): {avg_reward:.2f}, "
                  f"Value Loss: {avg_value_loss:.4f}")
        
        # Save checkpoint every 500 episodes
        if (episode + 1) % 500 == 0:
            checkpoint_path = f'pong_baseline_checkpoint_ep{episode + 1}.pth'
            torch.save({
                'episode': episode + 1,
                'policy_state_dict': policy.state_dict(),
                'value_state_dict': value_net.state_dict(),
                'policy_optimizer_state_dict': policy_optimizer.state_dict(),
                'value_optimizer_state_dict': value_optimizer.state_dict(),
                'episode_rewards': episode_rewards,
            }, checkpoint_path)
            print(f"  → Checkpoint saved: {checkpoint_path}")
    
    env.close()
    return episode_rewards


def evaluate_policy(env_name, policy, num_episodes, action_map):
    """Evaluate trained policy over multiple episodes"""
    env = gym.make(env_name)
    eval_rewards = []
    
    for episode in range(num_episodes):
        state, _ = env.reset()
        state = preprocess(state)
        prev_frame = None
        
        episode_reward = 0
        done = False
        
        while not done:
            # Use frame difference
            cur_frame = state
            if prev_frame is not None:
                state_input = cur_frame - prev_frame
            else:
                state_input = np.zeros_like(cur_frame, dtype=np.float32)
            prev_frame = cur_frame
            
            state_tensor = torch.FloatTensor(state_input).to(device)
            
            with torch.no_grad():
                action_probs = policy(state_tensor)
                action = torch.argmax(action_probs).item()
            
            env_action = action_map[action]
            
            next_state, reward, terminated, truncated, _ = env.step(env_action)
            done = terminated or truncated
            
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


# ==================== Main Training Script ====================

def train_pong():
    """Train Pong-v5 with Value Function Baseline"""
    print("\n" + "="*70)
    print("Training Pong-v5 with Value Function Baseline (Actor-Critic)")
    print("="*70 + "\n")
    
    print("BASELINE EXPLANATION:")
    print("-" * 70)
    print("Baseline Type: Learned Value Function V(s)")
    print("Implementation: Actor-Critic with CNN architecture")
    print()
    print("How it works:")
    print("  1. Actor (Policy): CNN that outputs action probabilities")
    print("  2. Critic (Value): Separate CNN that estimates V(s)")
    print("  3. Advantage: A(s,a) = G_t - V(s_t)")
    print("  4. Policy gradient: ∇J = E[∇log π(a|s) * A(s,a)]")
    print("  5. Frame differencing captures motion for both networks")
    print()
    print("Benefits for Pong:")
    print("  - Reduces high variance from sparse/delayed rewards")
    print("  - Critic learns long-term value of game states")
    print("  - More stable training in complex visual environments")
    print("-" * 70 + "\n")
    
    # Hyperparameters (same as Part 1 for fair comparison)
    gamma = 0.99
    learning_rate = 0.0003  # Smaller LR improves stability for actor-critic on Atari
    num_episodes = 1000
    
    # Action mapping: policy outputs 0 or 1, map to RIGHT(2) or LEFT(3)
    action_map = [2, 3]
    
    # Initialize networks
    policy = PongPolicy(action_dim=2).to(device)
    value_net = PongValueNetwork().to(device)
    
    # Initialize optimizers
    policy_optimizer = optim.Adam(policy.parameters(), lr=learning_rate)
    value_optimizer = optim.Adam(value_net.parameters(), lr=learning_rate)
    
    print(f"Hyperparameters:")
    print(f"  Discount factor (γ): {gamma}")
    print(f"  Learning rate: {learning_rate}")
    print(f"  Episodes: {num_episodes}")
    print(f"  Action mapping: 0->RIGHT(2), 1->LEFT(3)")
    print(f"  Gradient clipping: max_norm=1.0")
    print(f"  Architecture: CNN (80x80) → Conv → Conv → FC → Output")
    print()
    
    # Train with periodic checkpointing
    print("Starting training (checkpoints saved every 500 episodes)...\n")
    print("Stability settings: entropy_coeff=0.01, lr=3e-4, value loss=Huber, returns normalized")
    episode_rewards = train_policy_gradient_baseline(
        'ALE/Pong-v5', policy, value_net, policy_optimizer,
        value_optimizer, gamma, num_episodes, action_map, entropy_coeff=0.01
    )
    
    print("\nTraining completed!")
    
    # Evaluate
    print("\nEvaluating trained policy...")
    eval_rewards = evaluate_policy(
        'ALE/Pong-v5', policy, num_episodes=500, action_map=action_map
    )
    
    # Plot results
    plot_results(episode_rewards, eval_rewards, 
                'Pong-v5 with Baseline', 'pong_baseline')
    
    # Save models
    torch.save({
        'policy_state_dict': policy.state_dict(),
        'value_state_dict': value_net.state_dict(),
    }, 'pong_baseline_policy.pth')
    print("\nModels saved as 'pong_baseline_policy.pth'")
    
    return policy, value_net, episode_rewards, eval_rewards


# ==================== Run Training ====================

if __name__ == "__main__":
    # Train Pong with baseline
    print("\nNote: Pong training will take significantly longer (may take hours)")
    print("Training with Actor-Critic for variance reduction.\n")
    
    pong_policy, pong_value, pong_train_rewards, pong_eval_rewards = train_pong()

