import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='PPO algorithm with optional features.')
    parser.add_argument('--lr_annealing', action='store_true', help='Enable learning rate annealing.')
    parser.add_argument('--adam_epsilon_annealing', action='store_true', help='Enable Adam epsilon annealing.')
    parser.add_argument('--value_loss_clipping', action='store_true', help='Enable value function loss clipping.')

    args = parser.parse_args()
    return args
