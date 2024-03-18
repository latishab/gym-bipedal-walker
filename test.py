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
        self.fc3 = nn.Linear(32, action_dim)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return torch.tanh(self.fc3(x))  # Assuming continuous action space

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


# Helper functions
def sample_action(state, actor):
    print("State before conversion:", state)  # Debugging line
    state = torch.FloatTensor(state).unsqueeze(0)
    action = actor(state).detach().numpy()[0]  # Getting the action from the actor network
    return action

# Convert list of tensors to a tensor
def to_tensor(lst):
    return torch.tensor(lst, dtype=torch.float)

def main():
    num_episodes = 10

    # Initialize the BipedalWalker-v3 environment
    env = gym.make("BipedalWalker-v3", render_mode="human")
    num_inputs  = env.observation_space.shape[0]
    num_outputs = env.action_space.shape[0]

    actor = Actor(num_inputs, num_outputs)
    critic = Critic(num_inputs)

    actor_optimizer = optim.Adam(actor.parameters(), lr=3e-4)
    critic_optimizer = optim.Adam(critic.parameters(), lr=3e-4)

    for episode in range(num_episodes):
        ppo_epochs = 4  # Number of times to iterate over the whole dataset for updating the policy
        mini_batch_size = 64  # Number of samples to use for each mini-batch update

        # Initial reset returns a tuple (state, {})
        state, _ = env.reset() # Unpack the tuple here to get just the state
        total_reward = 0
        done = False

        states, actions, rewards, next_states, dones = [], [], [], [], []

        while not done:
            print("Current state:", state)  # Debugging line
            action = sample_action(state, actor)
            next_state, reward, done, truncated, _ = env.step(action)
            
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            next_states.append(next_state)
            dones.append(done)

            state = next_state
            total_reward += reward

        ppo_update(actor, critic, actor_optimizer, critic_optimizer, 
                   states, actions, rewards, next_states, dones, ppo_epochs, mini_batch_size)
        print(f"Episode: {episode+1}, Total Reward: {total_reward}")

if __name__ == "__main__":
    main()
    env.close() # Close the environment