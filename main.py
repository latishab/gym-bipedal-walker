from ppo import PPO
from utils import make_env
from parser import parse_args
import gym

def train(timesteps, args):
    env = make_env()
    model = PPO(env, args)
    model.learn(total_timesteps=timesteps)
    model.save_model()

def test(num_episodes=10, seed=0):
    env = gym.make('BipedalWalker-v3', render_mode="human")
    model = PPO(env)
    model.load_model()

    for ep in range(num_episodes):
        observation, info = env.reset(seed=0)
        done = False

        while not done:
            action, _ = model.get_action(observation)
            action = action.squeeze()
            observation, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            if done:
                env.render()
                if "episode" in info.keys():
                    print(f"Episode {ep}: Episodic Return = {info['episode']['r']}, Episodic Length = {info['episode']['l']}")

    env.close()

if __name__ == '__main__':
    # Train the model
    args = parse_args()
    timesteps=3000000
    train(timesteps, args)

    # # Test the model
    # test()
