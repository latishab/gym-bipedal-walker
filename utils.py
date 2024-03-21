import torch
import matplotlib.pyplot as plt

def approximate_kl_divergence(new_log_probs, old_log_probs):
    # Calculate the log probability ratios
    # http://joschu.net/blog/kl-approx.html
    log_ratios = new_log_probs - old_log_probs
    k3 = (torch.exp(log_ratios) - 1) - log_ratios # (exp(log_ratios) - 1) - log_ratios
    
    approximate_kl = k3.mean() # The mean of k3 provides the approximate KL divergence
    
    return approximate_kl

def make_plot(x_param, y_param, x_label, y_label, title, file_name):
    # Check if x_param and y_param have the same length
    if len(x_param) != len(y_param):
        print(f"Cannot plot because the lengths of x_param ({len(x_param)}) and y_param ({len(y_param)}) do not match.")
        return  # Exit the function if lengths do not match
    
    plt.plot(x_param, y_param)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.savefig(file_name)  # save the plot


