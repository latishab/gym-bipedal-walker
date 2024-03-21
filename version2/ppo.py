import torch
import torch.nn as nn
from torch.optim import Adam
from torch.distributions import MultivariateNormal
import numpy as np

# import local functions
from version2.networks import Agent
from utils import approximate_kl_divergence, make_plot

class PPO:
    def __init__(self, env, render=False):
        # Extract env info
        self.env = env
        self.num_envs = len(env.envs)
        self.render = render
        self.device = "cpu"

        # Initialize actor and critic networks with custom architecture
        self.agent = Agent(env).to(self.device)

        self.__init__hyperparameters()

        # Initialize actor and critic optimizer
        self.optimizer = Adam(self.agent.parameters(), lr= self.lr, eps=1e-5)

    def __init__hyperparameters(self):
        # Default values for hyperparameters
        self.gamma = 0.99 # discount factor
        self.lam = 0.95 # lambda parameter for GAE
        self.clip = 0.2 # clip threshold
        self.lr = 0.01 # learning rate of optimizers
        self.entropy_coef = 0.01 # entropy coefficient
        self.c_coef = 0.5 # critic function coefficient
        self.max_grad_norm = 0.5 # global grad clipping threshold
        self.target_kl = 0.02  # KL Divergence threshold
        self.num_minibatches = 4 # Number of mini-batches for Mini-batch Update
        self.n_updates_per_iteration = 4 # number of epochs per iteration
        self.num_steps = 1600
        self.batch_size = self.num_steps * self.num_envs
        self.minibatch_size = self.batch_size // self.num_minibatches

        # to store data for plotting
        self.lr_history = []  # to store learning rate per episode
        self.total_loss_history = []  # to store actor loss per episode
        self.all_episode_rewards = []  # to store total rewards per episode

    # Saving the actor and critic so we can load it back
    def save_model(self, actor_path='ppo_actor.pth', critic_path='ppo_critic.pth'):
        torch.save(self.agent.actor.state_dict(), actor_path)
        torch.save(self.agent.critic.state_dict(), critic_path)
        print("Model saved successfully.")

    # Load a model
    def load_model(self, actor_path='ppo_actor.pth', critic_path='ppo_critic.pth'):
        self.agent.actor.load_state_dict(torch.load(actor_path))
        self.agent.critic.load_state_dict(torch.load(critic_path))
        print("Model loaded successfully.")
    
    def get_action(self, obs):
        obs = torch.tensor(obs, dtype=torch.float)  # Ensure obs is a tensor

        # Query the actor network for a mean action
        mean = self.agent.actor(obs)

        # Create Multivariate Normal Distribution
        dist = MultivariateNormal(mean, self.cov_mat)

        # Sample an action from the distribution and get its log prob
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action.detach().numpy(), log_prob.detach() 

    def compute_gae(self, rewards, values, next_value, next_done, dones):
        """Compute Generalized Advantage Estimation"""
        advantages = torch.zeros_like(rewards).to(self.device)
        returns = torch.zeros_like(rewards)
        last_gae_lam = 0
        for t in reversed(range(self.num_steps)):
            if t == self.num_steps - 1:
                next_non_terminal = 1.0 - next_done
                next_values = next_value
            else:
                next_non_terminal = 1.0 - dones[t + 1]
                next_values = values[t + 1]
            delta = rewards[t] + self.gamma * next_values * next_non_terminal - values[t]
            advantages[t] = last_gae_lam = delta + self.gamma * self.lam * next_non_terminal * last_gae_lam
        
        returns = advantages + values
        return advantages, returns

    def learn(self, total_timesteps):
        # Initialize storage setup [ALGO]
        obs = torch.zeros((self.num_steps, self.num_envs) + self.env.single_observation_space.shape).to(self.device)
        actions = torch.zeros((self.num_steps, self.num_envs) + self.env.single_action_space.shape).to(self.device)
        log_probs = torch.zeros((self.num_steps, self.num_envs)).to(self.device)
        rewards = torch.zeros((self.num_steps, self.num_envs)).to(self.device)
        dones = torch.zeros((self.num_steps, self.num_envs)).to(self.device)
        values = torch.zeros((self.num_steps, self.num_envs)).to(self.device)

        ep_rewards = np.zeros(self.num_envs)  # Track rewards accumulated in each environment
        ep_timesteps = np.zeros(self.num_envs, dtype=int)  # Track timesteps for each environment

        t_so_far = 0  # Total timesteps simulated so far
        next_obs = torch.Tensor(self.env.reset()).to(self.device)
        next_done = torch.zeros(self.num_envs).to(self.device)
        num_updates = total_timesteps // (self.batch_size)

        for update in range(1, num_updates + 1):
            # Learning rate annealing [Optimization]
            frac = 1.0 - (update - 1.0) / num_updates
            new_lr = self.lr * frac
            self.optimizer.param_groups[0]['lr'] = max(new_lr, 1e-5)
            self.lr_history.append((t_so_far, new_lr))

            for step in range(0, self.num_steps):
                t_so_far += 1 * self.num_envs
                obs[step] = next_obs
                print(f"The next_done shape is {next_done.shape}")
                print(f"The dones shape is {dones.shape}")
                dones[step] = next_done

                # Get the action and log probs [ALGO]
                with torch.no_grad():
                    action, logprob, _, value = self.agent.get_action_and_value(next_obs)
                    values[step] = value.flatten()
                actions[step] = action
                log_probs[step] = logprob

                # Execute the game and log data
                next_obs, reward, dones, info = self.env.step(action.cpu().numpy())
                rewards[step] = torch.tensor(reward).to(self.device).view(-1)
                next_obs, next_done = torch.Tensor(next_obs).to(self.device), torch.Tensor(dones).to(self.device)

                # Accumulate rewards and track per environment
                for i, rew in enumerate(reward):
                    ep_rewards[i] += rew
                    ep_timesteps[i] += 1  # Increment episode timestep for each environment

                    if dones[i] or (ep_timesteps[i] == self.num_steps):
                        ep_timesteps[i] = t_so_far  # Update timestep at which the episode ended for each env
                        all_episode_rewards.append((t_so_far, ep_rewards[i])) # Log total rewards per episode
                        if ep_rewards[i] >= 300:
                            print(f"Episode with total rewards of {ep_rewards[i]} found at timestep {ep_timesteps[i]}")

                        # Reset for next episode
                        ep_rewards[i] = 0  
                        ep_timesteps[i] = 0

            with torch.no_grad():
                next_value = self.agent.get_value(next_obs).reshape(1, -1)
                # Compute GAE (advantages, returns)
                advantages, returns = self.compute_gae(rewards, values, next_value, next_done, dones)

            # flatten the batch
            batch_obs = obs.reshape((-1,) + self.single_observation_space.shape)
            batch_logprobs = log_probs.reshape(-1)
            batch_actions = actions.reshape((-1,) + self.env.single_action_space.shape)
            batch_advantages = advantages.reshape(-1)
            batch_returns = returns.reshape(-1)
            batch_values = values.reshape(-1)

            batch_inds = np.arange(self.batch_size)
            clipfracs = []
            for _ in range(self.n_updates_per_iteration):
                np.random.shuffle(batch_inds)
                for start in range(0, self.batch_size, self.minibatch_size):
                    end = start + self.minibatch_size
                    mini_inds = batch_inds[start:end]

                    _, new_log_prob, entropy, newvalue = self.agent.get_action_and_value(batch_obs[mini_inds], batch_actions.long()[mini_inds])
                    mini_log_prob = batch_logprobs[mini_inds]
                    log_ratio = new_log_prob - mini_log_prob
                    ratios = torch.exp(log_ratio)

                    with torch.no_grad():
                        approx_kl = approximate_kl_divergence(new_log_prob, mini_log_prob) # approximate kl divergence
                        clipfracs += [((ratios - 1.0).abs() > self.clip).float().mean().item()] 

                    mini_advantages = batch_advantages[mini_inds]
                    mini_advantages = (mini_advantages - mini_advantages.mean()) / (mini_advantages.std() + 1e-8) # normalize advantage

                    # Actor/Policy loss
                    actor_loss1 = -mini_advantages * ratios
                    actor_loss2 = -mini_advantages * torch.clamp(ratios, 1 - self.clip, 1 + self.clip) # calculate clip surrogate objective
                    actor_loss = torch.max(actor_loss1, actor_loss2).mean()

                    # Clipped critic loss
                    newvalue = newvalue.view(-1)
                    critic_loss_unclipped = (newvalue - batch_returns[mini_inds]) ** 2
                    critic_clipped = batch_values[mini_inds] + torch.clamp(
                        newvalue - batch_values[mini_inds],
                        -self.clip,
                        self.clip,
                    ) # also clip the value loss
                    critic_loss_clipped = (critic_clipped - batch_returns[mini_inds]) ** 2
                    critic_loss_max = torch.max(critic_loss_unclipped, critic_loss_clipped)
                    critic_loss = 0.5 * critic_loss_max.mean() * self.c_coef

                    # Entropy regularization [Optimization]
                    entropy_loss = entropy.mean()
                    loss = actor_loss - self.ent_coef * entropy_loss + critic_loss * self.c_coef

                    # Gradient update [Optimzation]
                    self.optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.agent.parameters(), self.max_grad_norm)
                    self.optimizer.step()

                    # Append actor_loss for logging
                    self.total_loss_history.append((t_so_far, actor_loss.item()))

                # Check for early stopping due to KL divergence
                with torch.no_grad():
                    if approx_kl > self.target_kl:
                        break

        # After learning is done, plot the total rewards per timesteps
        cumulative_timesteps, all_episode_rewards = zip(*self.all_episode_rewards)
        make_plot(cumulative_timesteps,
                  all_episode_rewards,
                  'Timesteps', 'Total Rewards', 'Total Rewards over Timesteps', 'rewards_plot.png')

        # Plot the actor loss
        cumulative_timesteps, actor_loss_history = zip(*self.total_loss_history)
        make_plot(cumulative_timesteps, 
                  actor_loss_history, 
                  'Timesteps', 'Total Loss', 'Total Loss over Timesteps', 'actor_loss.png')

        # Plot the learning rate 
        cumulative_timesteps, lr_history = zip(*self.lr_history)
        make_plot(cumulative_timesteps,
                lr_history, 
                'Timesteps', 'Learning Rate', 'Learning Rate over Timesteps', 'learning_rate.png')

