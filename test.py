import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
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

def compute_gae(next_value, rewards, masks, values, gamma=0.99, tau=0.95):
    values = values + [next_value]
    gae = 0
    returns = []
    for step in reversed(range(len(rewards))):
        delta = rewards[step] + gamma * values[step + 1] * masks[step] - values[step]
        gae = delta + gamma * tau * masks[step] * gae
        returns.insert(0, gae + values[step])
    return returns

def ppo_iter(mini_batch_size, states, actions, log_probs, returns, advantage):
    batch_size = states.size(0)
    for _ in range(batch_size // mini_batch_size):
        rand_ids = np.random.randint(0, batch_size, mini_batch_size)
        yield states[rand_ids, :], actions[rand_ids, :], log_probs[rand_ids, :], returns[rand_ids, :], advantage[rand_ids, :]

def ppo_update(actor, critic, actor_optimizer, critic_optimizer, states, actions, rewards, next_states, dones, ppo_epochs, mini_batch_size, clip_param=0.2, gamma=0.99, tau=0.95):
    states = torch.stack(states)
    next_states = torch.stack(next_states)
    actions = torch.stack(actions)
    rewards = torch.tensor(rewards, dtype=torch.float32)
    dones = torch.tensor(dones, dtype=torch.float32)
    log_probs = torch.stack(log_probs)  # Assume log_probs are collected during data gathering

    # Calculate value estimates for states and next_states
    values = critic(states).detach().squeeze()
    next_values = critic(next_states).detach().squeeze()

    # Calculate GAE and returns
    advantages, returns = compute_gae(next_values, rewards, 1 - dones, values, gamma, tau)
    
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-10)  # Normalize advantages
    
    for _ in range(ppo_epochs):
        # Assuming ppo_iter yields mini-batches
        for state, action, old_log_prob, return_, adv in ppo_iter(mini_batch_size, states, actions, log_probs, returns, advantages):
            # Get current policies' log probabilities and value estimates for the given states
            dist = actor(state)
            value = critic(state).squeeze()
            
            # Assume dist outputs have method .log_prob() and .entropy()
            new_log_prob = dist.log_prob(action)
            entropy = dist.entropy().mean()
            
            # Calculating the ratio (pi_theta / pi_theta_old)
            ratios = torch.exp(new_log_prob - old_log_prob)
            
            # Calculating Surrogate Loss
            surr1 = ratios * adv
            surr2 = torch.clamp(ratios, 1.0 - clip_param, 1.0 + clip_param) * adv
            actor_loss = - torch.min(surr1, surr2).mean() - 0.01 * entropy  # Including entropy to encourage exploration
            
            # Finalizing critic loss
            critic_loss = F.mse_loss(return_, value)
            
            # Perform gradient update
            actor_optimizer.zero_grad()
            actor_loss.backward()
            actor_optimizer.step()
            
            critic_optimizer.zero_grad()
            critic_loss.backward()
            critic_optimizer.step()

# Helper function
def sample_action(state, actor):
    state = torch.FloatTensor(state).unsqueeze(0)
    with torch.no_grad():
        mean, log_std = actor(state)
        std = log_std.exp()
        normal_dist = torch.distributions.Normal(mean, std)
        action = normal_dist.sample() # Using Normal over Categorical because we are in continuous space
        log_prob = normal_dist.log_prob(action).sum(axis=-1, keepdim=True)  # Sum across action dimensions

    # Squeeze the unnecessary dimensions and convert to numpy array
    action = action.squeeze().numpy()  # This should adjust the shape correctly
    log_prob = log_prob.squeeze().numpy()  # Adjust if needed, depending on how you use log_prob
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
        states, actions, rewards, next_states, dones, log_probs = [], [], [], [], [], []

        while not done:
            print("Current state:", state)  # Debugging line
            action, log_prob = sample_action(state, actor)
            # Validation check - adjust or remove after confirming it works as expected
            assert action.shape == (4,), f"Expected action shape (4,), got {action.shape}"
            next_state, reward, done, truncated, _ = env.step(action)
            
            # Store data as tensors [Data Collection]
            states.append(state)
            actions.append(torch.FloatTensor(action))
            rewards.append(reward)
            next_states.append(torch.FloatTensor(next_state).unsqueeze(0))
            dones.append(float(done)) # Indicating whether each episode has ended
            log_probs.append(log_prob)  # Storing logarithm probabilities for each action

            state = torch.FloatTensor(next_state).unsqueeze(0)  # Prepare for the next iteration
            total_reward += reward

        # Convert rewards and dones to tensors
        rewards = torch.tensor(rewards, dtype=torch.float32)
        dones = torch.tensor(dones, dtype=torch.float32)

        ppo_update(actor, critic, actor_optimizer, critic_optimizer, 
                   states, actions, rewards, next_states, dones, ppo_epochs, mini_batch_size)
        print(f"Episode: {episode+1}, Total Reward: {total_reward}")

if __name__ == "__main__":
    main()
    env.close() # Close the environment