import gym
import numpy as np
from version2.ppo import PPO  

def make_env(env_id, seed=1):
    def thunk():
        env = gym.make(env_id)
        env.seed(seed)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        return env

    return thunk

def train(env_id, timesteps, num_envs=2):
    envs = gym.vector.SyncVectorEnv(
        [make_env(env_id) for i in range(num_envs)]
    )
    assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"
    model = PPO(envs)
    model.learn(total_timesteps=timesteps)
    model.save_model()

def test(env_id, num_envs):
    envs = gym.vector.SyncVectorEnv(
        [make_env(env_id) for i in range(num_envs)]
    )
    assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"
    model = PPO(envs)
    model.load_model()

    for episode in range(10):
        obs = envs.reset()
        done = [False]
        total_rewards = np.zeros(envs.num_envs)

        while not all(done):
            action, _ = model.get_action(obs)
            obs, rewards, done, info = envs.step(action)
            total_rewards += rewards

            if 'episode' in info[0].keys():
                print(f"Episode {episode + 1}: Episodic Return = {info[0]['episode']['r']}")

if __name__ == '__main__':
    env_id = "BipedalWalker-v3"
    timesteps = 3000000
    
    # Train the model
    train(env_id, timesteps, 2)

    # Test the model
    # test()
