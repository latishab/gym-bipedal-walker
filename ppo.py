import gym
import time

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.distributions import Normal

# import local functions
from networks import MLPActor, MLPCritic
from utils import approximate_kl_divergence, make_plot
import numpy as np

class PPO:
    def __init__(self, env, args, render=False):
        # Extract env info
        self.env = env
        self.obs_dim = env.observation_space.shape[0]
        self.act_dim = env.action_space.shape[0]
        self.render = render
        self.args = args

        # Initialize actor and critic networks
        self.actor = MLPActor(self.obs_dim, self.act_dim, std=0.01)
        self.critic = MLPCritic(self.obs_dim, 1, std=1.0)

        self.__init__hyperparameters()

        # Initialize actor and critic optimizer
        self.actor_optim = Adam(self.actor.parameters(), lr = self.lr_actor, eps=self.epsilon_start)
        self.critic_optim = Adam(self.critic.parameters(), lr = self.lr_critic, eps=self.epsilon_end)

    def __init__hyperparameters(self):
        # Default values for hyperparameters
        self.gamma = 0.99 # discount factor
        self.lam = 0.95 # lambda parameter for GAE
        self.clip = 0.2 # clip threshold
        self.lr_actor = 3e-4 # learning rate for actor
        self.lr_critic = 4e-4 # learning rate for critic
        self.entropy_coef = 0.0 # entropy coefficient
        self.max_grad_norm = 0.5 # global grad clipping threshold
        self.target_kl = 0.02  # KL Divergence threshold
        self.num_minibatches = 32 # Number of mini-batches for Mini-batch Update
        self.timesteps_per_batch = 4800
        self.max_timesteps_per_episode = 1600
        self.n_updates_per_iteration = 10 # number of epochs per iteration

        # for epsilon annealing
        self.epsilon_start = 1e-6
        self.epsilon_end = 1e-4

        # to store data for plotting
        self.lr_history = []  # to store learning rate per episode
        self.actor_loss_history = []  # to store actor loss per episode
        self.episode_data = []  # to store total rewards per episode

    # Saving the actor and critic so we can load it back
    def save_model(self, actor_path='ppo_actor.pth', critic_path='ppo_critic.pth'):
        torch.save(self.actor.state_dict(), actor_path)
        torch.save(self.critic.state_dict(), critic_path)
        print("Model saved successfully.")

    # Load the model
    def load_model(self, actor_path='ppo_actor.pth', critic_path='ppo_critic.pth'):
        self.actor.load_state_dict(torch.load(actor_path))
        self.critic.load_state_dict(torch.load(critic_path))
        print("Model loaded successfully.")

    def get_action(self, obs):
        # Query the actor network for mean and std of the action distribution
        mean, std = self.actor(obs)

        # Create a Normal Distribution and sample an action
        dist = Normal(mean, std)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(-1)

        return action.detach().numpy(), log_prob.detach() # We detach the computational graphs (because its a tensor)

    def compute_gae(self, rewards, values, dones):
        batch_advantages = [] # List to store computed advantages for each step

        # Iterate over each episode's rewards, values, dones flags
        for ep_rewards, ep_vals, ep_dones in zip(rewards, values, dones):
            advantages = []
            last_advantage = 0

            # Calculate episode advantage in reversed order (from last step to first)
            for t in reversed(range(len(ep_rewards))): 
                if t + 1 < len(ep_rewards):
                    # Calculate the temporal difference (TD) error for current step
                    delta = ep_rewards[t] + self.gamma * ep_vals[t+1] * (1 - ep_dones[t+1]) - ep_vals[t]
                else: 
                    # Special case at the boundary (last step)
                    delta = ep_rewards[t] - ep_vals[t]
                
                # Calculate Generalized Advantage Estimation (GAE) for the current step
                advantage = delta + self.gamma * self.lam * (1 - ep_dones[t]) * last_advantage
                last_advantage = advantage  # Update the last advantage for the next step
                advantages.insert(0, advantage)  # Insert advantage at the beginning of the list
            
            # Extend the batch_advantages list with advantages computed for the current episode
            batch_advantages.extend(advantages)

        return torch.tensor(batch_advantages, dtype=torch.float32)
    
    def compute_rtgs(self, batch_rewards):
        # The rewards-to-go (rtg) per episode per batch to return.
        # The shape will be (num steps per episode)
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
    
    def evaluate(self, batch_obs, batch_acts):
        # Query critic network for a value of V for each obs in batch_obs
        V = self.critic(batch_obs).squeeze()

        # Calculate the log probabilities of batch actions using most recent actor network
        mean, std = self.actor(batch_obs)
        dist = Normal(mean, std)
        log_probs = dist.log_prob(batch_acts).sum(-1)
        entropy = dist.entropy()

        return V, log_probs, entropy

    def learn(self, total_timesteps):
        t_so_far = 0 # Timesteps simulated so far
        ep_count = 0 # Initialize episode counter

        print(f"The total timesteps are: {total_timesteps}")
        while t_so_far < total_timesteps:
            batch_obs, batch_acts, batch_log_probs, batch_rtgs, batch_lens, batch_rewards, batch_vals, batch_dones, ep_count = self.rollout(total_timesteps, ep_count, t_so_far)

            # Calculate advantage using GAE and normalize it
            A_k = self.compute_gae(batch_rewards, batch_vals, batch_dones)
            V = self.critic(batch_obs).squeeze()
            batch_rtgs = A_k + V.detach()   
            A_k = (A_k - A_k.mean()) / (A_k.std() + 1e-10)

            # Calculate how many steps we have collected this batch
            t_so_far += np.sum(batch_lens)

            # Print average episodic returns
            _, episode_returns = zip(*self.episode_data)
            if len(episode_returns) >= 100:
                # Calculate the average of the last 100 episodes
                last_100_avg = np.mean(episode_returns[-100:])
                print(f"Average episodic return for the last 100 episodes: {last_100_avg:.2f}")
            else:
                # If we haven't reached 100 episodes yet, calculate the average of all episodes so far
                overall_avg = np.mean(episode_returns)
                print(f"Average episodic return for all episodes so far: {overall_avg:.2f}")
                                       
            step = batch_obs.size(0)
            inds = np.arange(step)
            minibatch_size = step // self.num_minibatches
            loss = []

            for _ in range(self.n_updates_per_iteration):
                if self.args.lr_annealing:
                    # Perform learning rate annealing [Optimization]
                    frac = (t_so_far-1.0)/total_timesteps # calculate the fraction of total timesteps completed
                    new_lr_actor = self.lr_actor * (1.0 - frac) # calculate the new learning rate for actor
                    new_lr_critic = self.lr_critic * (1.0 - frac) # calculate the new learning rate for critic

                    # Make sure the new learning rate didn't go below 0
                    new_lr_actor = max(new_lr_actor, 0.0)
                    new_lr_critic = max(new_lr_critic, 0.0)
                    for param_group in self.actor_optim.param_groups:
                        param_group["lr"] = new_lr_actor
                    for param_group in self.critic_optim.param_groups:
                        param_group["lr"] = new_lr_critic
                    self.lr_history.append((t_so_far, new_lr_actor))

                if self.args.adam_epsilon_annealing:
                    # Epsilon annealing
                    self.critic_optim.param_groups[0]["eps"] = max(self.epsilon_start - t_so_far * (self.epsilon_start - self.epsilon_end) / total_timesteps, self.epsilon_end)

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

                    # Entropy regularization [Optimization]
                    entropy_loss = entropy.mean()
                    actor_loss = actor_loss - self.entropy_coef * entropy_loss 

                    if self.args.value_loss_clipping:
                        # Calculate clipped value function loss
                        estimated_value = self.critic(mini_obs).squeeze()
                        critic_loss1 = torch.square(estimated_value - mini_rtgs)
                        estimated_value_clipped = mini_rtgs + torch.clamp(self.critic(mini_obs).squeeze() - mini_rtgs, -self.clip, self.clip)
                        critic_loss2 = torch.square(estimated_value_clipped - mini_rtgs)
                        critic_loss = 0.5 * (torch.maximum(critic_loss1, critic_loss2)).mean()
                    else:
                        # Standard value function loss calculation without clipping
                        V = self.critic(mini_obs).squeeze()
                        critic_loss = nn.MSELoss()(V, mini_rtgs)

                    # Backpropagation for actor network
                    self.actor_optim.zero_grad()
                    actor_loss.backward(retain_graph=True)
                    nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm) # global gradient clipping [Optimization]
                    self.actor_optim.step()

                    # Backpropagation for critic network
                    self.critic_optim.zero_grad()
                    critic_loss.backward(retain_graph=True)
                    nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm) # global gradient clipping [Optimization]
                    self.critic_optim.step()

                    # Store the losses
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

    def rollout(self, total_timesteps, ep_count, t_so_far):
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

            obs, _ = self.env.reset(seed=0)
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
                action = action.squeeze()
                val = self.critic(obs)
                obs, rewards, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated

                # Collect reward, action, log_prob
                ep_rewards.append(rewards)
                ep_vals.append(val.flatten())
                batch_acts.append(action)
                batch_log_probs.append(log_prob)

                if done:
                    if "episode" in info.keys():  # Check if 'episode' key exists in info
                        episodic_return = info['episode']['r']
                        episodic_length = info['episode']['l']
                        curr_timestep = t_so_far + np.sum(batch_lens) + ep_t + 1
                        
                        if episodic_return >= 300:
                            print(f"Found an episode with a total reward of {episodic_return} at timestep {curr_timestep}.")
                        # print(f"Episode {ep_count} finished after {ep_t+1} timesteps with reward: {episodic_return}")

                        # store the episodic returns
                        self.episode_data.append((curr_timestep, episodic_return))

                        print(f"Episode {ep_count} finished after {episodic_length} timesteps with reward: {episodic_return} | global_steps={curr_timestep}")
                        ep_count += 1 
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