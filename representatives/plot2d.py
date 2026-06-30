import numpy as np
import matplotlib.pyplot as plt
import json
from collections import defaultdict
import os
import re
import glob
import pandas as pd
import math
from pathlib import Path
plt.rcParams.update({'font.size': 9})
import argparse

class ResultFolder:
    def __init__(self, path, parameter_name='phi_values'):
        self._folder_path = path
        # self._results = defaultdict(lambda: [])
        self._results = None
        self._max_episode = 0
        self._parameter_name = parameter_name
        self._parse_folder()

    def _parse_file(self, file_obj, start, fmt):
        if self._parameter_name in ["states", "actions_gr"]:
            fmt = "numpy"
        if fmt == "json":
            data_list = json.load(file_obj)
        elif fmt == "numpy":
            data_list = np.load(file_obj)
        if type(data_list) != dict and type(data_list) != np.lib.npyio.NpzFile:
            if self._results is None:
                self._results = defaultdict(lambda: [])
            episode_index = start
            for episode_data in data_list:
                self._results[episode_index] = episode_data
                episode_index += 1
            if episode_index > self._max_episode:
                self._max_episode = episode_index
        else:
            if self._results is None:
                self._results = defaultdict(lambda: defaultdict(lambda: []))
            for subparam, subparam_data_list in data_list.items():
                episode_index = start
                for episode_data in subparam_data_list:
                    self._results[subparam][episode_index] = episode_data
                    episode_index += 1
                if episode_index > self._max_episode:
                    self._max_episode = episode_index

    def _parse_folder(self, path=None):
        cur_path = path if path else self._folder_path
        files_parsed = 0
        with os.scandir(cur_path) as files:
            for file in files:
                if file.is_dir(follow_symlinks=False):
                    self._parse_folder(file.path)
                    continue
                if not file.is_file():
                    continue
                match = re.search(f'({self._parameter_name})_([0-9]+)-([0-9]+)\.np[y|z]', file.name)
                if not match:
                    continue
                with open(file.path, 'rb') as f:
                    self._parse_file(f, fmt="numpy", start=int(match[2]))
                    files_parsed += 1
                # print(f'parsed file {file.name}')
        if files_parsed == 0:
            with os.scandir(cur_path) as files:
                for file in files:
                    if file.is_dir(follow_symlinks=False):
                        self._parse_folder(file.path)
                        continue
                    if not file.is_file():
                        continue
                    match = re.search(f'({self._parameter_name})_([0-9]+)-([0-9]+)\.json', file.name)
                    if not match:
                        continue
                    if os.stat(file.path).st_size > 0:
                        with open(file.path, 'r') as f:
                            self._parse_file(f, fmt="json", start=int(match[2]))
                            files_parsed += 1
                    # print(f'parsed file {file.name}')
        # print(f'parsed folder {cur_path}')

def load_data(result_paths, params):
    folders = defaultdict(dict)
    values = defaultdict(dict)
    first_values = defaultdict(dict)
    evals = defaultdict(dict)
    evals_full = defaultdict(dict)
    for i, result_path in enumerate(result_paths):
        for param in params:
            if param == "phi_values_full":
                folders[i]["phi_values"] = ResultFolder(result_path, parameter_name="phi_values")
                res = folders[i]["phi_values"]._results
            else:
                folders[i][param] = ResultFolder(result_path, parameter_name=param)
                res = folders[i][param]._results
            if res is None:
                print(result_path)
            if param == "phi_values":
                if "phi" in res.keys() or "phi_values" in res.keys():
                    for subparam, res_subparam in res.items():
                        if subparam == "phi" or subparam == "phi_values":
                            max_episode = len(res_subparam)
                            values[i][param] = np.array([res_subparam[episode][-1] for episode in range(0, max_episode)])
                            first_values[i][param] = np.array([res_subparam[episode][0] for episode in range(0, max_episode)])
                else:
                    max_episode = len(res)
                    values[i][param] = [res[episode][-1] for episode in range(0, max_episode)]
                    first_values[i][param] = [res[episode][0] for episode in range(0, max_episode)]
                print(f"{i}, {result_path}: {max_episode} episodes")
            elif param == "phi_values_full":
                if "phi" in res.keys() or "phi_values" in res.keys():
                    for subparam, res_subparam in res.items():
                        if subparam == "phi" or subparam == "phi_values":
                            max_episode = len(res_subparam)
                            values[i][param] = np.array([res_subparam[episode] for episode in range(0, max_episode)])
                else:
                    max_episode = len(res)
                    values[i][param] = [res[episode] for episode in range(0, max_episode)]
            else:
                for subparam, res_subparam in res.items():
                    max_episode = len(res_subparam)
                    values[i][subparam] = [res_subparam[episode] for episode in range(0, max_episode)]
        eval_filename = os.path.join(result_path, "results", "eval.json")
        eval_full_filename = os.path.join(result_path, "results", "eval_full.json")
        eval_data = None
        eval_data_full = None
        if os.path.isfile(eval_filename):
            with open(eval_filename, "r") as f:
                eval_data = json.load(f)
        evals[i] = eval_data

        if os.path.isfile(eval_full_filename):
            with open(eval_full_filename, "r") as f:
                eval_data_full = json.load(f)
        evals_full[i] = eval_data_full
    return folders, values, first_values, evals, evals_full

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

def load_preds(result_path, data_path):
    folders, values_test, first_values, evals, evals_full = load_data([data_path], ["states", "actions_gr"])
    data_input_test = values_test[0]
    n_msg_iter = len(data_input_test.keys()) - 2
    actions_test = np.array(data_input_test['actions_gr'])
    states_test = {}
    for i in range(n_msg_iter + 1):
        states_test[i] = np.array(data_input_test[f"{i}"])
    n_agents = actions_test.shape[-1]

    X = {}
    for iteration in range(n_msg_iter + 1):
        X[iteration] = states_test[iteration].reshape(-1, n_agents, states_test[iteration].shape[-1])
    y = actions_test.reshape(-1, n_agents)
    n_points = len(y)

    pred = {i: {} for i in range(n_msg_iter + 1)}
    confidences = {i: {} for i in range(n_msg_iter + 1)}
    confidences_raw = {i: {} for i in range(n_msg_iter + 1)}
    confident = {i: {} for i in range(n_msg_iter + 1)}
    covered = {i: {} for i in range(n_msg_iter + 1)}
    for iteration in range(n_msg_iter + 1):
        for agent in range(n_agents):
            pred_data = np.load(os.path.join(result_path, f"pred_a{agent:02}_i{iteration}.npz"))
            pred[iteration][agent] = pred_data['pred']
            confidences[iteration][agent] = pred_data['confidences']
            confidences_raw[iteration][agent] = pred_data['confidences_raw']
            confident[iteration][agent] = pred_data['confident']
            covered[iteration][agent] = pred_data['covered']

        pred[iteration] = np.vstack([pred[iteration][agent] for agent in range(n_agents)])
        confidences[iteration] = np.vstack([confidences[iteration][agent] for agent in range(n_agents)])
        confidences_raw[iteration] = np.vstack([confidences_raw[iteration][agent] for agent in range(n_agents)])
        confident[iteration] = np.vstack([confident[iteration][agent] for agent in range(n_agents)])
        covered[iteration] = np.vstack([covered[iteration][agent] for agent in range(n_agents)])
    return X, y, pred, confidences, confidences_raw, confident, covered

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Read two files via named parameters')
    parser.add_argument('-i', '--input', default='plot2d_input.json', help='Path to json file with experiment paths')
    parser.add_argument('--test', default='test4', help='Path to testing dataset')
    parser.add_argument('-o', '--out', default='plot2d', help='File name prefix for output images')
    args = parser.parse_args()
    paths_json_filepath = args.input
    with open(paths_json_filepath, 'r', encoding='utf-8') as f:
        paths_json_data = json.load(f)
    exp_folders = paths_json_data["paths"]
    data_path_test = args.test
    out_filename = args.out
    out_filename_noext, out_filename_ext = os.path.splitext(out_filename)
    exp_names = [get_exp_name(Path(exp_folder).name) for exp_folder in exp_folders]
    results = []
    for exp_folder, exp_name in zip(exp_folders, exp_names):
        _, y, pred, confidences_notraw, confidences, confident, covered = \
            load_preds(exp_folder, data_path_test)
        n_points = len(y)
        n_agents = y.shape[-1]
        n_msg_iter = len(pred) - 1
        if "knn" in exp_name.lower() or "centroids" in exp_name.lower():
            thetas = np.array([0.])
        else:
            thetas = np.array([0.])
        for theta in thetas:
            accuracy_stats = {}
            f1_stats = {}
            coverage_stats = {}
            for iteration in range(n_msg_iter + 1):
                y_pred = pred[iteration].T
                if theta == 0.:
                    idx_predicted = (confidences[iteration] > theta).T
                else:
                    idx_predicted = (confidences[iteration] >= theta).T
                accuracy = np.zeros(n_agents)
                for agent in range(n_agents):
                    idx = idx_predicted[:, agent]
                    if np.sum(idx) > 0:
                        accuracy[agent] = np.mean(y[idx, agent] == y_pred[idx, agent])
                    else:
                        accuracy[agent] = float('nan')
                accuracy2 = accuracy[~np.isnan(accuracy)]
                coverage = np.sum(idx_predicted, axis=0) / len(y)
                if len(accuracy2) > 0:
                    accuracy_stats[iteration] = np.mean(accuracy2)
                else:
                    accuracy_stats[iteration] = 0.
                coverage_stats[iteration] = np.mean(coverage)
            result = [exp_name, theta]
            for iteration in range(n_msg_iter + 1):
                result.append(accuracy_stats[iteration])
                result.append(coverage_stats[iteration])
            results.append(result)

    columns = ["experiment", "theta"]
    for iteration in range(n_msg_iter + 1):
        columns.append(f"accuracy_iter{iteration}")
        columns.append(f"coverage_iter{iteration}")
    df = pd.DataFrame(data=results, columns=columns)

    dfc = df.copy()
    dfc['percentile'] = dfc['experiment'].str.extract(r'p=(.+)%').fillna(0).astype(float)
    cm = plt.get_cmap('jet')
    dfc_epsnet = dfc[dfc.experiment.str.contains("epsilon_net")]
    dfc_knn = dfc[dfc.experiment.str.contains("kNN")]
    dfc_centroids = dfc[dfc.experiment.str.contains("centroids")]

    for iteration in range(n_msg_iter + 1):
        cbar_vmin = math.log10(0.0002)
        cbar_vmax = math.log10(70)
        plt.figure(figsize=(7,4), dpi=300)
        ax = plt.gca()

        try:
            if iteration < 3:
                idx_annotate = [1, 2]
                bbox_alpha = 0.5
            else:
                idx_annotate = [1]
                bbox_alpha = 0.6
            for x, y in zip(dfc_epsnet[f'accuracy_iter{iteration}'][idx_annotate], dfc_epsnet[f'coverage_iter{iteration}'][idx_annotate]):
                ax.annotate(f"({x:.3f}, {y:.2f})", (x, y), ha='right', bbox=dict(facecolor='white', edgecolor='none', alpha=bbox_alpha))
        except Exception as e:
            pass

        for x, y in zip(dfc_knn[f'accuracy_iter{iteration}'], dfc_knn[f'coverage_iter{iteration}']):
                ax.annotate(f"({x:.3f}, {y:.2f})", (x, y), ha='left', bbox=dict(facecolor='white', edgecolor='none', alpha=0.4))

        for x, y in zip(dfc_centroids[f'accuracy_iter{iteration}'], dfc_centroids[f'coverage_iter{iteration}']):
            ax.annotate(f"({x:.3f}, {y:.2f})", (x, y), ha='left', bbox=dict(facecolor='white', edgecolor='none', alpha=0.4))

        sc = plt.scatter(dfc_epsnet[f'accuracy_iter{iteration}'], dfc_epsnet[f'coverage_iter{iteration}'], c=np.log10(np.maximum(dfc_epsnet.percentile, 1e-20)),
                         vmin=cbar_vmin, vmax=cbar_vmax, s=50, cmap=cm,
                         marker='D',
                         edgecolor='black',
                         label="PεN")
        sc2 = plt.scatter(dfc_knn[f'accuracy_iter{iteration}'], dfc_knn[f'coverage_iter{iteration}'],
                          color="black", s=60, marker='<', edgecolor='black',
                          label="kNN")
        sc3 = plt.scatter(dfc_centroids[f'accuracy_iter{iteration}'], dfc_centroids[f'coverage_iter{iteration}'],
                          color="purple", marker='o', edgecolor='black', s=60,
                          label="centroids")

        plt.ylabel("coverage")
        plt.xlabel("accuracy")
        cbar = plt.colorbar(sc)
        cbar_ticks = cbar.get_ticks()
        cbar_ticks = [cbar_vmin] + [x for x in cbar_ticks if cbar_vmin < x < cbar_vmax] + [cbar_vmax]
        cbar.ax.set_yticks(cbar_ticks)
        cbar.ax.set_yticklabels([f"{10**x:.5}" for x in cbar_ticks])
        cbar.set_label('distance percentile, %', rotation=90)
        plt.legend()
        plt.grid()
        plt.title(f"iteration {iteration}")
        out_filename_cur = f"{out_filename_noext}_iter{iteration}"
        plt.savefig(out_filename_cur, bbox_inches='tight')
        print(f"saved accuracy/coverage 2d plot for iteration {iteration} to {out_filename_cur}.png")
        plt.close()
