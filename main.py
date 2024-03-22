from ppo import PPO
from utils import make_env
import gym

def train(timesteps):
    env = make_env()
    model = PPO(env)
    model.learn(total_timesteps=timesteps)
    model.save_model()

def test(seed=0):
    env = gym.make('BipedalWalker-v3', render_mode="human")
    model = PPO(env)
    model.load_model()

    observation, info = env.reset(seed=0)
    for ep in range(1000):
        observation, reward, terminated, truncated, info = env.step(env.action_space.sample())
        print(info)
        if terminated or truncated:
            observation, info = env.reset()
            
        print(f"Episode {ep}: Episodic Return = {info['episode']['r']}, Episodic Length = {info['episode']['l']}")

if __name__ == '__main__':
    # Train the model
    timesteps=2000000
    train(timesteps)

    # # Test the model
    # test(seed=0)
