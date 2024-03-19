import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import copy
import matplotlib.pyplot as plt
import numpy as np
import gym

class Actor(nn.Module):
    def __init__(self, input_dim, action_dim):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.mean = nn.Linear(32, action_dim)
        self.log_std = nn.Linear(32, action_dim)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mean = torch.tanh(self.mean(x))  # Assuming continuous action space
        log_std = self.log_std(x)  # Log standard deviation
        log_std = torch.clamp(log_std, min=-20, max=2)  # Clamp for numerical stability
        return mean, log_std

class Critic(nn.Module):
    def __init__(self, input_dim):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

def compute_gae(next_value, rewards, masks, values, gamma=0.99, tau=0.95, normalize=True):
    # Convert PyTorch tensors to NumPy arrays for easier manipulation
    next_value_np = next_value.detach().cpu().numpy()
    rewards_np = rewards.detach().cpu().numpy()
    masks_np = masks.detach().cpu().numpy()
    values_np = values.detach().cpu().numpy()

    deltas = [r + gamma * (1 - d) * nv - v for r, d, nv, v in zip(rewards_np, masks_np, next_value_np, values_np)]
    deltas = np.array(deltas)  # Convert list to NumPy array

    # Initialize GAEs as a copy of deltas and perform the backward iteration
    gaes = copy.deepcopy(deltas)
    for t in reversed(range(len(deltas) - 1)):
        gaes[t] = gaes[t] + (1 - masks_np[t]) * gamma * tau * gaes[t + 1]

    # Compute the targets for value updates
    targets = gaes + values_np

    # Normalize GAEs if required
    if normalize:
        gaes = (gaes - gaes.mean()) / (gaes.std() + 1e-8)

    # Convert back to PyTorch tensors before returning
    gaes_torch = torch.tensor(gaes, dtype=torch.float32)
    targets_torch = torch.tensor(targets, dtype=torch.float32)

    # Ensure the tensors are in the correct shape
    gaes_torch = gaes_torch.view(-1, 1)
    targets_torch = targets_torch.view(-1, 1)

    return gaes_torch, targets_torch

def ppo_iter(mini_batch_size, states, actions, log_probs, returns, advantages):
    batch_size = states.size(0)
    for i in range(0, batch_size, mini_batch_size):
        start, end = i, min(i + mini_batch_size, batch_size)
        if advantages.nelement() == 0:
            print("Warning: Advantages tensor is empty. Skipping mini-batch.")
            continue
        yield states[start:end], actions[start:end], log_probs[start:end], returns[start:end], advantages[start:end]

def ppo_update(actor, critic, actor_optimizer, critic_optimizer, states, actions, rewards, next_states, dones, log_probs, ppo_epochs, mini_batch_size, clip_param=0.2, gamma=0.99, tau=0.95):

    # Ensure all inputs are tensors with the correct dimensions
    states = torch.stack(states).float().squeeze() if isinstance(states, list) else states.float().squeeze()
    next_states = torch.stack(next_states) if isinstance(next_states, list) else next_states.float().squeeze()
    actions = torch.stack(actions).float().squeeze() if isinstance(actions, list) else actions.float().squeeze()
    log_probs = torch.stack(log_probs).float().squeeze() if isinstance(log_probs, list) else log_probs.float().squeeze()
    rewards = torch.tensor(rewards, dtype=torch.float32)
    dones = torch.tensor(dones, dtype=torch.float32)

    # Calculate value estimates for states and next_states
    values = critic(states).squeeze()
    next_values = critic(next_states).squeeze()

    # Calculate GAE and returns
    advantages, returns = compute_gae(next_values, rewards, 1 - dones, values, gamma, tau)

    print(f"Advantages shape after compute_gae: {advantages.shape}")
    print(f"Returns shape after compute_gae: {returns.shape}")
    
    for _ in range(ppo_epochs):
        actor_optimizer.zero_grad()
        critic_optimizer.zero_grad()  # Zero gradients for both optimizers
        for idx in range(0, len(states), mini_batch_size):
            minibatch_indices = slice(idx, idx + mini_batch_size)
            minibatch_states = states[minibatch_indices]
            minibatch_actions = actions[minibatch_indices]
            minibatch_old_log_probs = log_probs[minibatch_indices]
            minibatch_rewards = rewards[minibatch_indices]
            minibatch_dones = dones[minibatch_indices]
            minibatch_next_states = next_states[minibatch_indices]

            # Recompute values for the current mini-batch
            minibatch_values = critic(minibatch_states).squeeze()
            minibatch_next_values = critic(minibatch_next_states).squeeze()

            # Recompute GAE and returns for the mini-batch
            minibatch_advantages, minibatch_returns = compute_gae(minibatch_next_values, minibatch_rewards, 1 - minibatch_dones, minibatch_values, gamma, tau)
            if minibatch_advantages.dim() == 0:
                minibatch_advantages = minibatch_advantages.unsqueeze(0)  # Ensure correct dimensionality

            # Compute new log probabilities, entropy, and ratio
            mean, log_std = actor(minibatch_states)
            dist = torch.distributions.Normal(mean, log_std.exp())
            minibatch_new_log_probs = dist.log_prob(minibatch_actions).sum(-1)
            ratio = torch.exp(minibatch_new_log_probs - minibatch_old_log_probs)

            # Compute surrogate loss
            surr1 = ratio * minibatch_advantages
            surr2 = torch.clamp(ratio, 1.0 - clip_param, 1.0 + clip_param) * minibatch_advantages
            actor_loss = -torch.min(surr1, surr2).mean()

            # Compute critic loss. Inputs needs to be a tensor instead a list
            critic_loss = F.mse_loss(minibatch_returns.squeeze(1), minibatch_values)

            # Combine losses for backpropagation
            total_loss = actor_loss + 0.5 * critic_loss - 0.01 * dist.entropy().mean()
            total_loss.backward()
        
        # Step optimizers after accumulating gradients
        actor_optimizer.step()
        critic_optimizer.step()

    print("Update complete.")

# Helper function
def sample_action(state, actor):
    state = torch.FloatTensor(state).unsqueeze(0)
    with torch.no_grad():
        mean, log_std = actor(state)
        std = log_std.exp()
        normal_dist = torch.distributions.Normal(mean, std)
        action = normal_dist.sample() # Using Normal over Categorical because we are in continuous space
        log_prob = normal_dist.log_prob(action).sum(axis=-1, keepdim=True)  # Sum across action dimensions

    # Keep action and log_prob as tensors for now
    action = action.squeeze() # Squeeze unnecessary dimensions
    return action, log_prob

# Convert list of tensors to a tensor
def to_tensor(lst):
    return torch.tensor(lst, dtype=torch.float)

def main():
    # Initialize ppo_epochs, mini_batch_size, and num_episodes
    ppo_epochs = 4  # Number of times to iterate over the whole dataset for updating the policy
    mini_batch_size = 64  # Number of samples to use for each mini-batch update
    num_episodes = 10

    # Initialize the BipedalWalker-v3 environment
    env = gym.make("BipedalWalker-v3", render_mode="human")
    num_inputs  = env.observation_space.shape[0]
    num_outputs = env.action_space.shape[0]

    # Actor and Critic
    actor = Actor(num_inputs, num_outputs)
    critic = Critic(num_inputs)
    actor_optimizer = optim.Adam(actor.parameters(), lr=3e-4)
    critic_optimizer = optim.Adam(critic.parameters(), lr=3e-4)

    for episode in range(num_episodes):
        # Initial reset returns a tuple (state, {})
        state, _ = env.reset() # Unpack the tuple here to get just the state
        state = torch.FloatTensor(state).unsqueeze(0)  # Convert numpy array to PyTorch tensor
        total_reward = 0
        done = False

        # Initialize lists for states, actions, rewards, next_states, dones, and log_probs [Data Collection]
        episode_rewards = []
        states, actions, rewards, next_states, dones, log_probs = [], [], [], [], [], []

        while not done:
            action, log_prob = sample_action(state, actor)
            action = action.squeeze().detach().cpu().numpy()
            # Validation check - adjust or remove after confirming it works as expected
            assert action.shape == (4,), f"Expected action shape (4,), got {action.shape}"
            next_state, reward, done, truncated, _ = env.step(action)
            
            # Store data as tensors [Data Collection]
            states.append(state)
            actions.append(torch.FloatTensor(action).unsqueeze(0))
            rewards.append(reward)
            next_states.append(torch.FloatTensor(next_state).unsqueeze(0))
            dones.append(float(done)) # Indicating whether each episode has ended
            log_probs.append(log_prob)  # Storing logarithm probabilities for each action

            state = torch.FloatTensor(next_state).unsqueeze(0)  # Prepare for the next iteration
            total_reward += reward

        # Convert lists to tensors
        rewards = torch.tensor(rewards).clone().detach().float()
        dones = torch.tensor(dones).clone().detach().float()
        log_probs = torch.stack(log_probs) 

        ppo_update(actor, critic, actor_optimizer, critic_optimizer, 
                   states, actions, rewards, next_states, dones, log_probs, ppo_epochs, mini_batch_size)
        print(f"Episode: {episode+1}, Total Reward: {total_reward}")
        episode_rewards.append(total_reward) # Append total_reward to the list for the current episode
    
    # After all episodes, plot the rewards
    plt.plot(episode_rewards)
    plt.title("Episode Rewards Over Time")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.savefig('rewards_plot.png')

    env.close() # Close the environment

if __name__ == "__main__":
    main()