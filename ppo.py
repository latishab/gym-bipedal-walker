import gym
import time

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.distributions import MultivariateNormal

# import local functions
from networks import FeedForwardNN
from utils import approximate_kl_divergence, make_plot

import matplotlib.pyplot as plt
import numpy as np

class PPO:
    def __init__(self, env, render=False):
        # Extract env info
        self.env = env
        self.obs_dim = env.observation_space.shape[0]
        self.act_dim = env.action_space.shape[0]
        self.render = render

        # Initialize actor and critic networks
        self.actor = FeedForwardNN(self.obs_dim, self.act_dim, std=0.01)
        self.critic = FeedForwardNN(self.obs_dim, 1, std=1.0)

        self.__init__hyperparameters()

        # Initialize actor and critic optimizer
        self.actor_optim = Adam(self.actor.parameters(), lr = self.lr, eps=1e-5)
        self.critic_optim = Adam(self.critic.parameters(), lr = self.lr, eps=1e-5)

        # Create a variable for matrix, as well as covariance matrix
        self.cov_var = torch.full(size=(self.act_dim,), fill_value=0.5)
        self.cov_mat = torch.diag(self.cov_var)

    def __init__hyperparameters(self):
        # Default values for hyperparameters
        self.gamma = 0.99 # discount factor
        self.lam = 0.95 # lambda parameter for GAE
        self.clip = 0.2 # clip threshold
        self.lr = 0.01 # learning rate of optimizers
        self.entropy_coef = 0.01 # entropy coefficient
        self.max_grad_norm = 0.5 # global grad clipping threshold
        self.target_kl = 0.02  # KL Divergence threshold
        self.num_minibatches = 4 # Number of mini-batches for Mini-batch Update
        self.timesteps_per_batch = 4800
        self.max_timesteps_per_episode = 1600
        self.n_updates_per_iteration = 4 # number of epochs per iteration

        # to store data for plotting
        self.lr_history = []  # to store learning rate per episode
        self.actor_loss_history = []  # to store actor loss per episode
        self.episode_data = []  # to store total rewards per episode

    # Saving the actor and critic so we can load it back
    def save_model(self, actor_path='ppo_actor.pth', critic_path='ppo_critic.pth'):
        torch.save(self.actor.state_dict(), actor_path)
        torch.save(self.critic.state_dict(), critic_path)
        print("Model saved successfully.")

    def load_model(self, actor_path='ppo_actor.pth', critic_path='ppo_critic.pth'):
        self.actor.load_state_dict(torch.load(actor_path))
        self.critic.load_state_dict(torch.load(critic_path))
        print("Model loaded successfully.")

    def get_action(self, obs):
        # Query the actor network for a mean action
        # Same thing as calling self.actor.forward(obs)
        mean = self.actor(obs)

        # Create Multivariate Normal Distribution
        dist = MultivariateNormal(mean, self.cov_mat)

        # Sample an action from the distribution and get its log prob
        action = dist.sample()
        log_prob = dist.log_prob(action)

        return action.detach().numpy(), log_prob.detach() # we detach the computational graphs (because its a tensor)

    def compute_gae(self, rewards, values, dones):
        batch_advantages = [] # List to store computed advantages for each timestep

        # Iterate over each episode's rewards, values, dones flags
        for ep_rewards, ep_vals, ep_dones in zip(rewards, values, dones):
            advantages = []
            last_advantage = 0

            # Calculate episode advantage in reversed order (from last timestep to first)
            for t in reversed(range(len(ep_rewards))): 
                if t + 1 < len(ep_rewards):
                    # Calculate the temporal difference (TD) error for the current timestep
                    delta = ep_rewards[t] + self.gamma * ep_vals[t+1] * (1 - ep_dones[t+1]) - ep_vals[t]
                else: 
                    # Special case at the boundary (last timestep)
                    delta = ep_rewards[t] - ep_vals[t]
                
                # Calculate Generalized Advantage Estimation (GAE) for the current timestep
                advantage = delta + self.gamma * self.lam * (1 - ep_dones[t]) * last_advantage
                last_advantage = advantage  # Update the last advantage for the next timestep
                advantages.insert(0, advantage)  # Insert advantage at the beginning of the list
            
            # Extend the batch_advantages list with advantages computed for the current episode
            batch_advantages.extend(advantages)

        return torch.tensor(batch_advantages, dtype=torch.float32)

    def learn(self, total_timesteps):
        t_so_far = 0 # Timesteps simulated so far
        ep_count = 0 # Initialize episode counter

        print(f"The total timesteps are: {total_timesteps}")
        while t_so_far < total_timesteps:
            batch_obs, batch_acts, batch_log_probs, batch_rtgs, batch_lens, batch_rewards, batch_vals, batch_dones, ep_count = self.rollout(total_timesteps, ep_count)

            # Calculate advantage using GAE and normalize it
            A_k = self.compute_gae(batch_rewards, batch_vals, batch_dones)
            V = self.critic(batch_obs).squeeze()
            batch_rtgs = A_k + V.detach()   
            A_k = (A_k - A_k.mean()) / (A_k.std() + 1e-10)

            # Calculate how many timesteps and all episode rewards we collected this batch
            t_so_far += np.sum(batch_lens)
            for ep_rewards in batch_rewards:
                ep_reward_sum = np.sum(ep_rewards)
                # Check if the episode's total reward is 300 or more
                if ep_reward_sum >= 300:
                    print(f"Found an episode with a total reward of {ep_reward_sum} at timestep {t_so_far}.")
                self.episode_data.append((t_so_far, ep_reward_sum)) # update episode data for plotting

            # Inside the learning loop, after updating all_episode_rewards
            _, episode_rewards = zip(*self.episode_data)
            if len(episode_rewards) >= 100:
                # Calculate the average of the last 100 episodes
                last_100_avg = np.mean(episode_rewards[-100:])
                print(f"Average episodic return for the last 100 episodes: {last_100_avg:.2f}")
            else:
                # If we haven't reached 100 episodes yet, calculate the average of all episodes so far
                overall_avg = np.mean(episode_rewards)
                print(f"Average episodic return for all episodes so far: {overall_avg:.2f}")
                                       
            step = batch_obs.size(0)
            inds = np.arange(step)
            minibatch_size = step // self.num_minibatches
            loss = []

            for _ in range(self.n_updates_per_iteration):
                # Perform learning rate annealing [Optimization]
                frac = (t_so_far-1.0)/total_timesteps # calculate the fraction of total timesteps completed
                new_lr = self.lr * (1.0 - frac) # calculate the new learning rate based on the fraction completed

                # Make sure the new learning rate didn't go below 0
                new_lr = max(new_lr, 0.0)
                for param_group in self.actor_optim.param_groups:
                    param_group["lr"] = new_lr
                for param_group in self.critic_optim.param_groups:
                    param_group["lr"] = new_lr
                self.lr_history.append((t_so_far, new_lr))

                # Mini-batch update [Optimization]
                np.random.shuffle(inds) # Shuffling the index
                for start in range(0, step, minibatch_size):
                    end = start + minibatch_size
                    idx = inds[start:end]

                    # Extract data at the sampled indices
                    mini_obs = batch_obs[idx]
                    mini_acts = batch_acts[idx]
                    mini_log_prob = batch_log_probs[idx]
                    mini_advantage = A_k[idx]
                    mini_rtgs = batch_rtgs[idx]

                    # Calculate V_{phi, k}, pi_theta(a_t | s_t) and entropy
                    V, curr_log_probs, entropy = self.evaluate(mini_obs, mini_acts)

                    # Calculate ratio (pi current policy / pi old policy)
                    ratios = torch.exp(curr_log_probs - mini_log_prob)
                    approx_kl = approximate_kl_divergence(curr_log_probs, mini_log_prob)

                    # Calculate clipped surrogate objective
                    surr1 = ratios * mini_advantage # r(t)A
                    surr2 = torch.clamp(ratios, 1-self.clip, 1+self.clip) * mini_advantage # clip(r(t), 1-e, 1+e)A

                    # Calculate actor loss
                    actor_loss = (-torch.min(surr1, surr2)).mean()
                    critic_loss = nn.MSELoss()(V, mini_rtgs)

                    # Entropy regularization [Optimization]
                    entropy_loss = entropy.mean()
                    actor_loss = actor_loss - self.entropy_coef * entropy_loss 

                    # Calculate gradients and perform backward propagation for actor network
                    self.actor_optim.zero_grad()
                    actor_loss.backward(retain_graph=True)
                    nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm) # gradient clipping [Optimization]
                    self.actor_optim.step()

                    # Calculate gradients and perform backward propagation for critic network
                    self.critic_optim.zero_grad()
                    critic_loss.backward(retain_graph=True)
                    nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm) # gradient clipping [Optimization]
                    self.critic_optim.step()

                    loss.append(actor_loss.detach())
                    self.actor_loss_history.append((t_so_far, actor_loss.item()))

                # Early stop if approx. KL divergence is above the target
                if approx_kl > self.target_kl:
                    break 

        # After learning is done, plot the total rewards per timesteps
        cumulative_timesteps, all_episode_rewards = zip(*self.episode_data)
        make_plot(cumulative_timesteps,
                  all_episode_rewards,
                  'Timesteps', 'Total Rewards', 'Total Rewards over Timesteps', 'rewards_plot.png', 'benchmarks/rewards')

        # Plot the actor loss
        cumulative_timesteps, actor_loss_history = zip(*self.actor_loss_history)
        make_plot(cumulative_timesteps, 
                  actor_loss_history, 
                  'Timesteps', 'Actor Loss', 'Actor Loss over Timesteps', 'actor_loss.png', 'benchmarks/losses')

        # Plot the learning rate 
        cumulative_timesteps, lr_history = zip(*self.lr_history)
        make_plot(cumulative_timesteps,
                lr_history, 
                'Timesteps', 'Learning Rate', 'Learning Rate over Timesteps', 'learning_rate.png', 'benchmarks/learning_rates')

    def evaluate(self, batch_obs, batch_acts):
        # Query critic network for a value of V for each obs in batch_obs
        V = self.critic(batch_obs).squeeze()

        # Calculate the log probabilities of batch actions using most recent actor network
        # This part is similar to get_action()
        mean = self.actor(batch_obs)
        dist = MultivariateNormal(mean, self.cov_mat)
        log_probs = dist.log_prob(batch_acts)

        return V, log_probs, dist.entropy()
    
    def compute_rtgs(self, batch_rewards):
        # The rewards-to-go (rtg) per episode per batch to return.
        # The shape will be (num timesteps per episode)
        batch_rtgs = []

        # Iterate through each episode backwards to maintain same order
        # in batch_rgts
        for ep_rewards in reversed(batch_rewards):
            discounted_reward = 0 # The discounted reward so far

            for reward in reversed(ep_rewards):
                discounted_reward = reward + discounted_reward * self.gamma
                batch_rtgs.insert(0, discounted_reward)
        
        # Convert the rewards-to-go to a tensor
        batch_rtgs = torch.tensor(batch_rtgs, dtype=torch.float)

        return batch_rtgs

    def rollout(self, total_timesteps, ep_count):
        """The agent follows its policy, interacts with the environment, and observes outcomes. 
        It's a "rollout" because you're simulating the path an agent takes in an environment 
        from start to finish for each episode."""
        # Batch data
        batch_obs = [] # batch observations
        batch_acts = [] # batch actions
        batch_log_probs = [] # log probs of each action
        batch_rewards = [] # batch rewards
        batch_rtgs = [] # batch rewards-to-go
        batch_lens = [] # episodic lengths in batch
        batch_vals = []
        batch_dones = []
        t = 0 # Initialize timestep counter

        # Number of timesteps so far for this batch
        while (t < self.timesteps_per_batch) and (t <= total_timesteps):
            # episodic data
            ep_rewards = [] # rewards collected per episode
            ep_vals = [] # state values collected per episode
            ep_dones = [] # done flag collected per episode

            obs, _ = self.env.reset()
            done = False

            for ep_t in range(self.max_timesteps_per_episode):
                # If render is specified, render the environment
                if self.render:
                    self.env.render()
                # Track done flag of the current state
                ep_dones.append(done)

                t += 1; # Increment timesteps for this batch so far

                # Collect observation
                batch_obs.append(obs)

                action, log_prob = self.get_action(obs)
                val = self.critic(obs)
                obs, rewards, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated

                # Collect reward, action, log_prob
                ep_rewards.append(rewards)
                ep_vals.append(val.flatten())
                batch_acts.append(action)
                batch_log_probs.append(log_prob)

                if done:
                    ep_reward_sum = np.sum(ep_rewards)
                    ep_count += 1 
                    print(f"Episode {ep_count} finished after {ep_t+1} timesteps with reward: {ep_reward_sum}")
                    break
            
            # Collect episodic lengths and rewards
            batch_lens.append(ep_t + 1) # plus 1 because timestep starts from 0
            batch_rewards.append(ep_rewards)
            batch_vals.append(ep_vals)
            batch_dones.append(ep_dones)

        # Reshape data as tensors in the shape specified before returning
        batch_obs = torch.tensor(batch_obs, dtype=torch.float)
        batch_acts = torch.tensor(batch_acts, dtype=torch.float)
        batch_log_probs = torch.tensor(batch_log_probs, dtype=torch.float)

        # Compute the rewards-to-go
        batch_rtgs = self.compute_rtgs(batch_rewards)

        return batch_obs, batch_acts, batch_log_probs, batch_rtgs, batch_lens, batch_rewards, batch_vals, batch_dones, ep_count