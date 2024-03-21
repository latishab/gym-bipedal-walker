import torch
import matplotlib.pyplot as plt
import os
import re

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
    
    plt.plot(x_param, y_param)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    
    # Save the plot with the generated filename
    plt.savefig(file_path)