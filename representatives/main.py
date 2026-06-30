import argparse
import numpy as np
import os
import sys
import time
import json
from collections import defaultdict
from pathlib import Path
import matplotlib.pyplot as plt
from experiment_runner import *
from types import SimpleNamespace
from data_loader import DataLoader
import copy


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='Epsilon-Net classifier for MAROH data analysis',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Usage examples:
  python3 main.py --config config.yaml
  python3 main.py --config config.yaml --theta 0.2 # override parameter
        '''
    )

    parser.add_argument('--config', type=str, default="config.yaml", help='Path to YAML configuration file')
    parser.add_argument('--data-path', type=str, help='Path to data')
    parser.add_argument('--data-path-test', type=str, help='Path to test data (if not planning to split data from data-path into train and test)')
    parser.add_argument('--mode', type=str, choices=['single', 'full'], help='Execution mode')
    parser.add_argument('--agent', type=int, help='Agent ID (single mode)')
    parser.add_argument('--iteration', type=int, help='Iteration number (single mode)')
    parser.add_argument('--classifier', type=str, help='Classifier')
    parser.add_argument('--searcher', type=str, help='Spatial search structure')
    parser.add_argument('--delta', type=float, help='Reliability parameter [0,1)')
    parser.add_argument('--theta', type=float, help='Confidence threshold [0,1]')
    parser.add_argument('--metric', type=str, choices=['euclidean', 'manhattan', 'chebyshev', 'cosine'], help='Metric')
    parser.add_argument('--train-ratio', type=float, help='Fraction of data for training')
    parser.add_argument('--max-test-points', type=int, help='Maximum test points')
    parser.add_argument('--use-acceleration', action='store_true', help='Use acceleration')
    parser.add_argument('--buffer-size', type=int, help='Buffer size')
    parser.add_argument('--output-dir', type=str, help='Directory for results')
    parser.add_argument('--verbose', action='store_true', help='Verbose output')
    parser.add_argument('--seed', type=int, help='Random seed')
    parser.add_argument('--save-plots', action='store_true', help='Save plots')
    parser.add_argument('--save-pred', action='store_true', help='Save all predictions on test set')
    parser.add_argument('--save-reprs', action='store_true', help='Save representative sets')
    parser.add_argument('--reprs-path', type=str, help='Directory from which EpsilonNet representatives will be loaded, instead of training')
    parser.add_argument('--update', action='store_true', help='')
    parser.add_argument('--update-buffer-length', type=int, help='')
    parser.add_argument('--max-updates', type=int, help='')
    parser.add_argument('--dist-percentile', type=float, help='Distance percentile')
    args = parser.parse_args()
    return args


def load_config(config_path):
    """Load configuration from YAML file"""
    if not config_path or not os.path.exists(config_path):
        return {}

    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)
    return {}

def merge_config_with_args(config_dict, args):
    """
    Merge config and command line arguments.
    Command line arguments have priority.
    """
    args_dict = {k: v for k, v in vars(args).items()
                 if v is not None and k != 'config'}

    # Recursive dictionary update
    def deep_update(base, update):
        for key, value in update.items():
            if isinstance(value, dict) and key in base and isinstance(base[key], dict):
                deep_update(base[key], value)
            elif value is not None and value is not False:
                base[key] = value
        return base

    merged = copy.deepcopy(config_dict) if config_dict else {}
    return deep_update(merged, args_dict)

def dict_to_namespace(d):
    """Recursive conversion of dictionary to SimpleNamespace"""
    if isinstance(d, dict):
        return SimpleNamespace(**{k: dict_to_namespace(v) for k, v in d.items()})
    elif isinstance(d, list):
        return [dict_to_namespace(item) for item in d]
    return d


def validate_config(config):
    """Validate required config fields"""
    if not hasattr(config, 'data_path'):
        raise ValueError("data_path not specified. Set in config or via --data-path")

    if config.mode == 'single':
        if not hasattr(config, 'agent') or config.agent is None:
            raise ValueError("For single mode, agent must be specified")
        if not hasattr(config, 'iteration') or config.iteration is None:
            raise ValueError("For single mode, iteration must be specified")

    # Validate ranges
    if hasattr(config, 'delta') and (config.delta < 0 or config.delta >= 1):
        raise ValueError("delta must be in range [0, 1)")

    if hasattr(config, 'theta') and (config.theta < 0 or config.theta > 1):
        raise ValueError("theta must be in range [0, 1]")

    if hasattr(config, 'train_ratio') and (config.train_ratio <= 0 or config.train_ratio >= 1):
        raise ValueError("train_ratio must be in range (0, 1)")

    return config

def print_config(config):
    """Print configuration"""
    print("\n" + "=" * 60)
    print("EXPERIMENT CONFIGURATION")
    print("=" * 60)

    def print_namespace(ns, indent=0):
        for key, value in vars(ns).items():
            if key.startswith('_'):
                continue
            prefix = "  " * indent
            if isinstance(value, SimpleNamespace):
                print(f"{prefix}{key}:")
                print_namespace(value, indent + 1)
            else:
                print(f"{prefix}{key}: {value}")

    print_namespace(config)
    print("=" * 60 + "\n")


def setup_output_directory(args):
    """Create and configure output directory"""

    timestamp = time.strftime('%Y%m%d_%H%M%S')
    theta_str = "" if args.theta == 0. else f"_theta{args.theta}"
    if args.mode == 'single':
        exp_name = f"{args.classifier}_agent{args.agent}_iter{args.iteration}{theta_str}_p{args.dist_percentile}_{args.data_path}_{timestamp}"
    else:
        exp_name = f"{args.classifier}_full{theta_str}_p{args.dist_percentile}_{args.data_path}_{timestamp}"

    args.output_dir = os.path.join(args.output_dir, exp_name)

    # Create directory
    os.makedirs(args.output_dir, exist_ok=True)

    if args.verbose:
        print(f"\nResults will be saved to: {args.output_dir}")
    return args

def setup_experiment(args):
    """
    Full experiment setup:
    - Load and merge configs
    - Create output directory
    - Set seed
    - Prepare list of agents and iterations

    Args:
        args: command line arguments
        data: loaded data (needed for full mode)
        data_test: separately loaded test data (if path specified in config)
    """
    config_dict = load_config(args.config)
    merged_dict = merge_config_with_args(config_dict, args)
    config = dict_to_namespace(merged_dict)
    config = validate_config(config)
    config = setup_output_directory(config)
    data = DataLoader.load_sequence_data(config.data_path)
    data_test = None
    if hasattr(config, 'data_path_test'):
        data_test = DataLoader.load_sequence_data(config.data_path_test)
    np.random.seed(args.seed)

    # Prepare agents and iterations
    if config.mode == 'single':
        config.agents = [config.agent]
        config.iterations = [config.iteration]

        # Remove original fields to avoid confusion
        if hasattr(config, 'agent'):
            delattr(config, 'agent')
        if hasattr(config, 'iteration'):
            delattr(config, 'iteration')

    elif config.mode == 'full' and data is not None:
        all_agents = sorted(np.unique(data['agent']))
        all_iterations = sorted(set(
            it for a in all_agents
            for it in np.unique(data[data['agent'] == a]['iteration'])
        ))
        if hasattr(config, 'agent') and config.agent == -1:
            config.agents = [-1]
        else:
            config.agents = all_agents
        if hasattr(config, 'iteration') and config.iteration == -1:
            config.iterations = [-1]
        else:
            config.iterations = all_iterations

    if config.verbose:
        print_config(config)
    return data, data_test, config


def main():
    """Main function - program entry point"""

    # Prepare input configuration
    args = parse_arguments()
    try:
        data, data_test, config = setup_experiment(args)
        run_analysis(data, data_test, config)

        print("\n" + "=" * 60)
        print("Execution completed successfully!")
        print(f"Results saved to: {config.output_dir}")
        print("=" * 60)

    except FileNotFoundError as e:
        print(f"\nError: File or directory '{config.data_path}' not found!")
        print(f"Details: {e}")
        sys.exit(1)

    except Exception as e:
        print(f"\nError: {e}")
        if config.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
