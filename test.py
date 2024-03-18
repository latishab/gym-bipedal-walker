import gym
import random
import numpy as np

# Initialize the BipedalWalker-v3 environment
env = gym.make('BipedalWalker-v3', render_mode="human")

# Reset the environment to get the initial state
observation = env.reset()

def Random_games():
    # Each of this episode is its own game
    action_size = env.action_space.shape[0]

    for episode in range(10):
        env.reset()
        # this is each frame, up to 500...but we won't make it that far with random
        while True:
            # This will display the environment
            # Only display if you really want to see it
            # Takes much longer to display it
            env.render()

            # This will create a sample action in any environment
            # In this environment, the action can be any of one how in list 4, for example [0 1 0 0]
            action = np.random.uniform(-1.0, 1.0, size=action_size)

            # this executes the environment with an action, 
            # and returns the observation of the environment, 
            # the reward, if the env is over, and other info.
            observation, reward, terminated, truncated, info = env.step(action)
            
            # lets print everything in one line:
            print(reward, action)
            if terminated or truncated:
                break  # Exit the loop if the episode is done

        # Reset the environment at the end of each episode
        env.reset()

Random_games()

# Close the environment
env.close()
