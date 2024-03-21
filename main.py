from ppo import PPO
import gym
import torch

def train(timesteps):
    env = gym.make('BipedalWalker-v3', render_mode="human")
    model = PPO(env)
    model.learn(total_timesteps=timesteps)
    model.save_model()

def test():
    env = gym.make('BipedalWalker-v3', render_mode="human")
    model = PPO(env)
    model.load_model()

    for episode in range(10):
        observation, _ = env.reset()
        done = False
        total_reward = 0
        
        while not done:
            env.render()
            observation = torch.tensor(observation, dtype=torch.float32)
            with torch.no_grad():
                action, _ = model.get_action(observation)
            observation, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_reward += reward
            
        print(f"Episode {episode + 1}: Episodic Return = {info['episode']['r']}")

if __name__ == '__main__':
    # Train the model
    timesteps=3000000
    train(timesteps)

    # # Test the model
    # test()
