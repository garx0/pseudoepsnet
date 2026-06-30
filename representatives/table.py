import yaml
import glob
from typing import List, Tuple, Dict, Set, Optional, Any, Callable, Union
from types import SimpleNamespace
import os
import json
import numpy as np
import argparse

def load_experiment(dirpath: str) -> Dict[str, Any]:
    """Loading one experiment from JSON"""
    filepaths = glob.glob(os.path.join(dirpath, "analysis_*.json"))
    assert len(filepaths) == 1, filepaths
    filepath = filepaths[0]
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)

def load_experiments_from_config(config: Any) -> List[Dict[str, Any]]:
    """
    Loading all experiments from comparison config

    Args:
        config: SimpleNamespace object with comparison config

    Returns:
        list of experiments with added data
    """
    experiments = []

    for exp_config in config.comparison["experiments"]:
        try:
            data = load_experiment(exp_config["path"])
            for i, r in enumerate(data['results']):
                r['metrics']['classified'] = r['metrics']['confident_classifications'] / r['metrics']['total_points']
                # data['results'][i]['metrics']['classified'] = r['metrics']['confident_classifications'] / r['metrics']['total_points']
            experiments.append({
                'name': exp_config["name"],
                'data': data,
                'results': data['results'],
                'params': data.get('params', {}),
                'classifier': data.get('classifier', 'unknown')
            })
            print(f"✓ Loaded: {exp_config['name']} ({len(data['results'])} results)")
        except Exception as e:
            print(f"✗ Error loading {exp_config['path']}: {e}")
    return experiments

def get_common_agents(experiments: List[Dict]) -> List[int]:
    """Getting list of agents present in all experiments"""
    agents_sets = [set(r['agent_id'] for r in exp['results']) for exp in experiments]
    if len(agents_sets) > 0:
        return sorted(set.intersection(*agents_sets))
    else:
        return []

def get_common_iterations(experiments: List[Dict]) -> List[int]:
    """Getting list of iterations present in all experiments"""
    iters_sets = [set(r['iteration_id'] for r in exp['results']) for exp in experiments]
    if len(iters_sets) > 0:
        return sorted(set.intersection(*iters_sets))
    else:
        return []

def save_comparison_table(experiments: List[Dict], save_dir: str, last_iteration=True):
    """Save results table"""
    """if last_iteration == False, experiments on last message exchange iteration aren't considered"""
    import csv

    csv_path = os.path.join(save_dir, 'comparison_summary.csv')
    agents = get_common_agents(experiments)
    iterations = get_common_iterations(experiments)
    if not last_iteration:
        print(last_iteration)
        iterations = iterations[:-1]

    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)

        columns = ['Classifier', 'Accuracy (mean±std)', 'Coverage (mean±std)', 'Compression (mean±std)',
            'Representatives (mean±std)', 'Antirepresentatives (mean±std)']
        for i in iterations:
            columns.extend([f'Accuracy (iteration {i}) (mean±std)',
                            f'Coverage (iteration {i}) (mean±std)',
                            f'Compression/coverage (iteration {i}) (mean±std)',
                            f'Representatives (iteration {i}) (mean±std)',
                            f'Antirepresentatives (iteration {i}) (mean±std)',
                        ])
        columns.extend(['Time (s)', 'Build time (s)', 'Evaluation time (μs)', 'Agents', 'Iterations'])
        writer.writerow(columns)

        for exp in experiments:
            train_data_size = exp['results'][0]['train_data_size']
            test_data_size = exp['results'][0]['train_data_size']
            acc = [r['metrics']['accuracy'] for r in exp['results']
                if last_iteration or r['iteration_id'] in iterations]
            cov = [r['metrics']['classified'] for r in exp['results']
                if last_iteration or r['iteration_id'] in iterations]
            comp = [r['metrics']['compression_ratio'] for r in exp['results']
                if last_iteration or r['iteration_id'] in iterations]
            times = [r['build_time'] + r['test_time'] for r in exp['results']
                if last_iteration or r['iteration_id'] in iterations]
            times_build = [r['build_time'] for r in exp['results']
                if last_iteration or r['iteration_id'] in iterations]
            times_test = [r['test_time'] / test_data_size * 1000000 for r in exp['results']
                if last_iteration or r['iteration_id'] in iterations]
            reprs = [r['representatives_count'] for r in exp['results']
                if last_iteration or r['iteration_id'] in iterations]
            print(reprs)
            antireprs = [r['class_distribution'].get("-1", 0) for r in exp['results']
                if last_iteration or r['iteration_id'] in iterations]

            row = [
                exp['name'],
                f"{np.mean(acc):.4f} ± {np.std(acc):.4f}",
                f"{np.mean(cov):.4f} ± {np.std(cov):.4f}",
                f"{np.mean(comp):.4f} ± {np.std(comp):.4f}",
                f"{np.mean(reprs):.1f} ± {np.std(reprs):.1f}",
                f"{np.mean(antireprs):.1f} ± {np.std(antireprs):.1f}",
            ]
            for i in iterations:
                acc_i = [r['metrics']['accuracy'] for r in exp['results'] if r['iteration_id'] == i]
                cov_i = [r['metrics']['classified'] for r in exp['results'] if r['iteration_id'] == i]
                comp_i = [r['metrics']['compression_ratio'] for r in exp['results'] if r['iteration_id'] == i]
                reprs_i = [r['representatives_count'] for r in exp['results'] if r['iteration_id'] == i]
                antireprs_i = [r['class_distribution'].get("-1", 0) for r in exp['results'] if r['iteration_id'] == i]
                row.extend([f"{np.mean(acc_i):.4f} ± {np.std(acc_i):.4f}",
                        f"{np.mean(cov_i):.4f} ± {np.std(cov_i):.4f}",
                        f"{np.mean(comp_i):.4f} ± {np.std(comp_i):.4f}",
                        f"{np.mean(reprs_i):.1f} ± {np.std(reprs_i):.1f}",
                        f"{np.mean(antireprs_i):.1f} ± {np.std(antireprs_i):.1f}",
                ])
            row.extend([f"{np.mean(times):.4f} ± {np.std(times):.4f}"])
            row.extend([f"{np.mean(times_build):.4f} ± {np.std(times_build):.4f}"])
            row.extend([f"{np.mean(times_test):.1f} ± {np.std(times_test):.1f}"])
            row.extend([f"{len(agents)}", f"{len(iterations)}"])
            writer.writerow(row)

    print(f"  → comparison_summary.csv")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Read two files via named parameters')
    parser.add_argument('-i', '--input', default='config_update.yaml', help='Path to yaml file with experiment paths and path for saving the table')
    parser.add_argument('--no-last-iter', action='store_true', help='Don''t consider experiments on last message exchange iteration')
    args = parser.parse_args()
    config_path = args.input
    with open(config_path, 'r', encoding='utf-8') as f:
        compare_dict = yaml.safe_load(f)

    config = SimpleNamespace(**compare_dict)

    # Make output dir
    save_dir = config.comparison["output_dir"]
    os.makedirs(save_dir, exist_ok=True)
    print(f"Table will be saved in {save_dir}")

    # Load experiments
    experiments = load_experiments_from_config(config)
    save_comparison_table(experiments, save_dir, last_iteration=not args.no_last_iter)

    print(f"Table saved in {save_dir}")
