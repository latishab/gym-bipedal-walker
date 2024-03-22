import torch
import matplotlib.pyplot as plt
import os
import re
import gym
import numpy as np

def make_env():
    render_mode = "rgb_array"
    env_id = "BipedalWalker-v3"
    env = gym.make(env_id, render_mode=render_mode)

    # Determine the new batch folder name
    video_folder_base = "videos"
    if not os.path.exists(video_folder_base):
        os.makedirs(video_folder_base)
    existing_batches = [d for d in os.listdir(video_folder_base) if os.path.isdir(os.path.join(video_folder_base, d)) and d.startswith("batch_")]
    if existing_batches:
        highest_batch_num = max([int(batch.split('_')[-1]) for batch in existing_batches])
        new_batch_num = highest_batch_num + 1
    else:
        new_batch_num = 1
    video_folder = os.path.join(video_folder_base, f"batch_{new_batch_num}")

    # Create the new batch directory if it does not exist
    if not os.path.exists(video_folder):
        os.makedirs(video_folder)

    # Setup environment with new video folder path
    env = gym.wrappers.RecordEpisodeStatistics(env)
    env = gym.wrappers.RecordVideo(
        env,
        video_folder=video_folder,  
        episode_trigger=lambda episode_id: episode_id % 100 == 0,
        name_prefix="bw-video"
    )
    env = gym.wrappers.ClipAction(env)
    env = gym.wrappers.NormalizeObservation(env)
    env = gym.wrappers.TransformObservation(env, lambda obs: np.clip(obs, -10, 10))
    env = gym.wrappers.NormalizeReward(env)
    env = gym.wrappers.TransformReward(env, lambda reward: np.clip(reward, -10, 10))

    return env

def approximate_kl_divergence(new_log_probs, old_log_probs):
    # Calculate the log probability ratios
    # http://joschu.net/blog/kl-approx.html
    log_ratios = new_log_probs - old_log_probs
    k3 = (torch.exp(log_ratios) - 1) - log_ratios # (exp(log_ratios) - 1) - log_ratios
    
    approximate_kl = k3.mean() # The mean of k3 provides the approximate KL divergence
    
    return approximate_kl

def make_plot(x_param, y_param, x_label, y_label, title, file_name, folder_path):
    # Check if x_param and y_param have the same length
    if len(x_param) != len(y_param):
        print(f"Cannot plot because the lengths of x_param ({len(x_param)}) and y_param ({len(y_param)}) do not match.")
        return  # Exit the function if lengths do not match
    
    # Ensure that the folder exists, if not, create it
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    
    # Generate a unique filename by appending a numerical suffix
    base_name, extension = os.path.splitext(file_name)

    # Extract numerical suffix from base filename if it exists
    match = re.match(r'^(.*)_(\d+)$', base_name)
    if match:
        base_name, num_suffix = match.groups()
        i = int(num_suffix) + 1
    else:
        num_suffix = ""
        i = 1

    while True:
        file_name = f"{base_name}_{i}{extension}"
        file_path = os.path.join(folder_path, file_name)
        if not os.path.exists(file_path):
            break
        i += 1

    plt.figure() # Start a fresh figure
    plt.plot(x_param, y_param)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.savefig(file_path) # Save the plot with the generated filename
    plt.clf() # Clear the figure