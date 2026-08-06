import numpy as np
import matplotlib.pyplot as plt
import json
import os
import glob
import pandas as pd
import math
import re
from pathlib import Path
plt.rcParams.update({'font.size': 9})
import argparse

def load_experiment(dirpath: str):
    """Loading one experiment from JSON"""
    filepaths = glob.glob(os.path.join(dirpath, "analysis_*.json"))
    assert len(filepaths) == 1, filepaths
    filepath = filepaths[0]
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)

def get_exp_name(exp_folder):
    m = re.match("(.+)_full(.*)_p([^_]+)_(.+)_2026.+", exp_folder)
    if m is None:
        exp_name = exp_folder
    else:
        algo, _, percentile, train_data = m.groups()
        if "knn" in algo.lower() or "centroids" in algo.lower():
            exp_name = f"{algo}"
        else:
            exp_name = f"{algo}, p={percentile}%"
    return exp_name

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Read two files via named parameters')
    parser.add_argument('-i', '--input', default='plot2d_input.json', help='Path to json file with experiment paths')
    parser.add_argument('-o', '--out', default='plot2d', help='File name prefix for output images')
    parser.add_argument('--no-last-iter', action='store_true', help='Don\'t consider experiments on last message exchange iteration')
    args = parser.parse_args()

    paths_json_filepath = args.input
    with open(paths_json_filepath, 'r', encoding='utf-8') as f:
        paths_json_data = json.load(f)
    exp_folders = paths_json_data["paths"]
    out_filename = args.out
    out_filename_noext, out_filename_ext = os.path.splitext(out_filename)

    # Load experiments
    experiments = []
    exp_names = []
    for exp_folder in exp_folders:
        try:
            exp_data = load_experiment(exp_folder)
            exp_name = get_exp_name(Path(exp_folder).name)
            experiments.append(exp_data)
            exp_names.append(exp_name)
            print(f"✓ Loaded: {exp_name} ({len(exp_data['results'])} results)")
        except Exception as e:
            print(f"✗ Error loading {exp_folder}: {e}")

    if not experiments:
        print("No experiments loaded!")
        exit(1)

    # Get common agents and iterations
    agents_sets = [set(r['agent_id'] for r in exp['results']) for exp in experiments]
    common_agents = sorted(set.intersection(*agents_sets)) if agents_sets else []

    iters_sets = [set(r['iteration_id'] for r in exp['results']) for exp in experiments]
    common_iterations = sorted(set.intersection(*iters_sets)) if iters_sets else []

    # Optionally remove last iteration
    if args.no_last_iter and len(common_iterations) > 0:
        common_iterations = common_iterations[:-1]
        print(f"Ignoring last iteration: {common_iterations[-1] if common_iterations else 'none'}")

    if not common_iterations:
        print("No common iterations found!")
        exit(1)

    # Extract data for each experiment
    results = []
    n_msg_iter = max(common_iterations)

    for exp_data, exp_name in zip(experiments, exp_names):
        # Filter results by common agents and iterations
        filtered_results = []
        for r in exp_data['results']:
            if r['agent_id'] in common_agents and r['iteration_id'] in common_iterations:
                filtered_results.append(r)

        if not filtered_results:
            continue

        # Calculate metrics per iteration
        for iteration in common_iterations:
            iter_results = [r for r in filtered_results if r['iteration_id'] == iteration]
            if not iter_results:
                continue

            accuracies = [r['metrics']['accuracy'] for r in iter_results]
            # Coverage (classified ratio)
            coverages = [r['metrics']['confident_classifications'] / r['metrics']['total_points']
                        for r in iter_results]
            compressions = [r['metrics']['compression_ratio']
                        for r in iter_results]

            mean_accuracy = np.mean(accuracies)
            mean_coverage = np.mean(coverages)
            mean_compression = np.mean(compressions)

            # Extract percentile from experiment name if exists
            percentile_match = re.search(r'p=([\d.]+)%', exp_name)
            if percentile_match:
                percentile = float(percentile_match.group(1))
            else:
                percentile = 0.0  # For kNN and centroids

            results.append({
                'experiment': exp_name,
                'iteration': iteration,
                'accuracy': mean_accuracy,
                'coverage': mean_coverage,
                'compression': mean_compression,
                'percentile': percentile,
            })

    # Create DataFrame
    df = pd.DataFrame(results)

    if df.empty:
        print("No data to plot!")
        exit(1)

    # Plot for each iteration
    for iteration in common_iterations:
        df_iter = df[df['iteration'] == iteration]
        if df_iter.empty:
            continue
        df_epsnet_iter = df_iter[df_iter['experiment'].str.contains("epsilon_net", case=False)]
        df_knn_iter = df_iter[df_iter['experiment'].str.contains("knn", case=False)]
        df_centroids_iter = df_iter[df_iter['experiment'].str.contains("centroids", case=False)]

        plt.figure(figsize=(7,4), dpi=300)
        ax = plt.gca()

        # Define color map for percentile
        cm = plt.get_cmap('jet')
        cbar_vmin = math.log10(1)
        cbar_vmax = math.log10(5000)

        # Plot epsilon_net points with percentile coloring
        if not df_epsnet_iter.empty:
            compressions_inv = 1 / df_epsnet_iter['compression'].values
            # Avoid log(0)
            compressions_inv_log = np.log10(np.maximum(compressions_inv, 1e-20))
            sc = plt.scatter(
                df_epsnet_iter['accuracy'],
                df_epsnet_iter['coverage'],
                c=compressions_inv_log,
                vmin=cbar_vmin, vmax=cbar_vmax,
                s=50, cmap=cm,
                marker='D',
                edgecolor='black',
                linewidth=0.5,
                label="PεN"
            )

            # Annotate points
            # if len(df_epsnet_iter) > 0:
            #     # idx_annotate = [1, 2] if iteration < 3 else [1]
            #     idx_annotate = [i for i in idx_annotate if i < len(df_epsnet_iter)]
            #     for i in idx_annotate:
            #         x = df_epsnet_iter.iloc[i]['accuracy']
            #         y = df_epsnet_iter.iloc[i]['coverage']
            #         ax.annotate(f"({x:.3f}, {y:.2f})", (x, y),
            #                    ha='left', fontsize=7,
            #                    bbox=dict(facecolor='white', edgecolor='none', alpha=0.5))

        # Plot kNN points
        if not df_knn_iter.empty:
            for _, row in df_knn_iter.iterrows():
                ax.annotate(f"({row['accuracy']:.3f}, {row['coverage']:.2f})",
                           (row['accuracy'], row['coverage']),
                           ha='left',
                           bbox=dict(facecolor='white', edgecolor='none', alpha=0.4))
            plt.scatter(
                df_knn_iter['accuracy'],
                df_knn_iter['coverage'],
                color="black", s=60, marker='x', edgecolor='black',
                label="k-NN"
            )

        # Plot centroids points
        if not df_centroids_iter.empty:
            for _, row in df_centroids_iter.iterrows():
                ax.annotate(f"({row['accuracy']:.3f}, {row['coverage']:.2f})",
                           (row['accuracy'], row['coverage']),
                           ha='left',
                           bbox=dict(facecolor='white', edgecolor='none', alpha=0.4))
            plt.scatter(
                df_centroids_iter['accuracy'],
                df_centroids_iter['coverage'],
                color="black", marker='*', edgecolor='black', s=60,
                label="Centroids"
            )

        plt.ylabel("coverage")
        plt.xlabel("accuracy")
        plt.grid(True)
        plt.legend()

        # Add colorbar for epsilon_net if present
        if not df_epsnet_iter.empty:
            cbar = plt.colorbar(sc)
            cbar_ticks = cbar.get_ticks()
            cbar_ticks = [cbar_vmin] + [x for x in cbar_ticks if cbar_vmin < x < cbar_vmax] + [cbar_vmax]
            cbar.ax.set_yticks(cbar_ticks)
            cbar.ax.set_yticklabels([f"{10**x:.5}" for x in cbar_ticks])
            cbar.set_label('1 / compression ratio', rotation=90)

        plt.title(f"iteration {iteration}")
        out_filename_cur = f"{out_filename_noext}_iter{iteration}"
        plt.savefig(out_filename_cur + '.png', bbox_inches='tight')
        print(f"saved accuracy/coverage 2d plot for iteration {iteration} to {out_filename_cur}.png")
        plt.close()

    dfs_iters = []
    for iteration in common_iterations:
        df_iter = df[df['iteration'] == iteration]
        df_iter = df_iter.rename(columns={
            'accuracy': f'accuracy (it.{iteration})',
            'coverage': f'coverage (it.{iteration})',
            'compression': f'compression (it.{iteration})'
        })
        df_iter = df_iter.drop(columns=['iteration'])
        dfs_iters.append(df_iter)
    df_merged = dfs_iters[0]
    for df_iter in dfs_iters[1:]:
        df_merged = pd.merge(
            df_merged,
            df_iter,
            on=['experiment', 'percentile'],
            how='outer'
        )
    out_filename_table = f"{out_filename_noext}_table.csv"
    df_merged.to_csv(out_filename_table)
    print(f"saved table to {out_filename_table}")
