import torch

def approximate_kl_divergence(new_log_probs, old_log_probs):
    # Calculate the log probability ratios
    # http://joschu.net/blog/kl-approx.html
    log_ratios = new_log_probs - old_log_probs
    k3 = (torch.exp(log_ratios) - 1) - log_ratios # (exp(log_ratios) - 1) - log_ratios
    
    approximate_kl = k3.mean() # The mean of k3 provides the approximate KL divergence
    
    return approximate_kl

