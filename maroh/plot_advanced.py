import numpy as np
import matplotlib.pyplot as plt
import json
from collections import Counter, defaultdict
import os
import re
import glob
import pandas as pd
from pathlib import Path
import argparse
plt.rcParams.update({'font.size': 9})

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

def get_config_value(exp_dir, key):
    filenames = glob.glob(os.path.join(exp_dir, "config*.yaml"))
    if len(filenames) > 1:
        raise ValueError(f"multiple configs in {exp_dir}: {filenames}")
    if len(filenames) == 0:
        raise ValueError(f"no configs in {exp_dir}")
    filename = filenames[0]
    return get_config_value_from_file(filename, key)

def get_config_value_from_file(filename, key):
    key = key.strip()
    if key.endswith(":"):
        key = key[:-1]
    with open(filename, "r") as f:
        for line in f:
            line = line.strip()
            if line.startswith(key):
                val = line[len(key):].strip()
                if val.startswith(":"):
                    val = val[1:].strip()
                    return val
                else:
                    continue
    return None

def get_ecmp_value(ecmp_dir, t=0, full=False):
    ecmp_dir = Path(ecmp_dir)
    ecmp_values = []
    i = 0
    for flows_filename in glob.glob(str(ecmp_dir / "flows*.json")):
        flows_path = Path(flows_filename)
        flows_name = flows_path.stem
        flows_ecmp_result_file = sorted(glob.glob(str(ecmp_dir / f"{flows_name}-ecmp-1-{t}*.npz")))[-1]
        arr = np.load(flows_ecmp_result_file)
        ecmp_values.append(arr['mlu'][0])
        i += 1
        if i > 0 and i % 200 == 0:
            print(f"read {i} files")
    if not full:
        return np.mean(ecmp_values)
    else:
        return np.array(ecmp_values)

def get_opt_value(opt_dir, t=0, full=False):
    opt_dir = Path(opt_dir)
    values = []
    i = 0
    for flows_filename in glob.glob(str(opt_dir / "flows*.json")):
        flows_path = Path(flows_filename)
        flows_name = flows_path.stem
        flows_result_file = sorted(glob.glob(str(opt_dir / f"{flows_name}-opt-{t}*.npy")))[-1]
        arr = np.load(flows_result_file)
        values.append(arr[0])
        i += 1
        if i > 0 and i % 200 == 0:
            print(f"read {i} files")
    if not full:
        return np.mean(values)
    else:
        return np.array(values)

def load_data(result_paths, params):
    folders = defaultdict(dict)
    values = defaultdict(dict)
    first_values = defaultdict(dict)
    evals = defaultdict(dict)
    evals_full = defaultdict(dict)
    update_period = {}
    for i, result_path in enumerate(result_paths):
        eval_data = None
        evals[i] = eval_data
        eval_data_full = None
        evals_full[i] = eval_data_full
        eval_filename = os.path.join(result_path, "results", "eval.json")
        eval_full_filename = os.path.join(result_path, "results", "eval_full.json")
        try:
            for param in params:
                if param == "phi_values_full":
                    folders[i]["phi_values"] = ResultFolder(result_path, parameter_name="phi_values")
                    res = folders[i]["phi_values"]._results
                else:
                    folders[i][param] = ResultFolder(result_path, parameter_name=param)
                    res = folders[i][param]._results
                if res is None:
                    # print(result_path)
                    pass
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

            if os.path.isfile(eval_filename):
                with open(eval_filename, "r") as f:
                    eval_data = json.load(f)
            evals[i] = eval_data
            if os.path.isfile(eval_full_filename):
                with open(eval_full_filename, "r") as f:
                    eval_data_full = json.load(f)
            evals_full[i] = eval_data_full
        except Exception as e:
            print(f"couldn't read jsons: {e}")
        evals[i] = eval_data
    return folders, values, first_values, evals, evals_full, update_period

def calc_accuracy(msgs, actions_gr, actions_gr_true, n_msg_iter):
    n_agents = msgs.shape[-1]
    y = {i: {agent: [] for agent in range(n_agents)} for i in range(n_msg_iter + 1)}
    y_pred = {i: {agent: [] for agent in range(n_agents)} for i in range(n_msg_iter + 1)}
    for i in range(len(msgs)):
        for agent in range(n_agents):
            m = msgs[i, agent]
            y[m][agent].append(actions_gr_true[i, agent])
            y_pred[m][agent].append(actions_gr[i, agent])
    for i in range(n_msg_iter + 1):
        for agent in range(n_agents):
            y[i][agent] = np.array(y[i][agent])
            y_pred[i][agent] = np.array(y_pred[i][agent])

    accuracy_stats = {}
    coverage_stats = {}
    accuracy_all = []
    coverage_all = []
    for i in range(n_msg_iter + 1):
        accuracy_arr = []
        coverage_arr = []
        for agent in range(n_agents):
            n_predicted = len(y[i][agent])
            coverage = n_predicted / len(msgs)
            if n_predicted > 0:
                accuracy = (y[i][agent] == y_pred[i][agent]).mean()
            else:
                accuracy = np.nan
            accuracy_arr.append(accuracy)
            coverage_arr.append(coverage)
        accuracy_arr = np.array(accuracy_arr)
        accuracy_arr_notnan = accuracy_arr[~np.isnan(accuracy_arr)]
        coverage_arr = np.array(coverage_arr)
        accuracy_stats[i] = accuracy_arr_notnan.mean() if len(accuracy_arr_notnan) > 0 else np.nan
        coverage_stats[i] = coverage_arr.mean()
        # print(f"iteration {i}: mean agent accuracy = {accuracy_arr_notnan.mean():.4f}, mean agent coverage = {coverage_arr.mean():.4f}")
        # print(f"by agents: accuracy: {accuracy_arr}")
        # print(f"           coverage: {coverage_arr}")
    agent_accuracy = []
    agent_coverage = []
    for agent in range(n_agents):
        n_predicted = sum([len(y[i][agent]) for i in range(n_msg_iter)])
        coverage = n_predicted / len(msgs)
        if n_predicted > 0:
            accuracy = np.concatenate([y[i][agent] == y_pred[i][agent] for i in range(n_msg_iter)]).mean()
        else:
            accuracy = np.nan
        agent_accuracy.append(accuracy)
        agent_coverage.append(coverage)
    agent_accuracy = np.array(agent_accuracy)
    agent_coverage = np.array(agent_coverage)
    return accuracy_stats, coverage_stats, np.mean(agent_accuracy[~np.isnan(agent_accuracy)]), np.mean(agent_coverage)

def get_neighbors(incoming_links, outcoming_links):
    n_agents = max(max(incoming_links), max(outcoming_links)) + 1
    neighbors_out = np.zeros((n_agents, n_agents), dtype=bool)
    neighbors_in = np.zeros((n_agents, n_agents), dtype=bool)
    # neighbors2_out = np.zeros((n_agents, n_agents), dtype=bool)
    # neighbors2_in = np.zeros((n_agents, n_agents), dtype=bool)
    for agent in range(n_agents):
        agents_range = np.arange(n_agents)
        neighbors_out_list = sorted(outcoming_links[incoming_links == agent])
        neighbors_out[agent] = np.isin(agents_range, neighbors_out_list, assume_unique=True)
        neighbors_in_list = sorted(incoming_links[outcoming_links == agent])
        neighbors_in[agent] = np.isin(agents_range, neighbors_in_list, assume_unique=True)
        # neighbors2_out_list = sorted(outcoming_links2[incoming_links2 == agent])
        # neighbors2_out[agent] = np.isin(agents_range, neighbors2_out_list, assume_unique=True)
        # neighbors2_in_list = sorted(incoming_links2[outcoming_links2 == agent])
        # neighbors2_in[agent] = np.isin(agents_range, neighbors2_in_list, assume_unique=True)

    # checking that neighbors_out and neighbors_in are consistent
    for agent, neighs in enumerate(neighbors_out):
        for agent2 in range(n_agents):
            assert neighbors_in[agent2][agent] == neighs[agent2]
    for agent, neighs in enumerate(neighbors_in):
        for agent2 in range(n_agents):
            assert neighbors_out[agent2][agent] == neighs[agent2]
    return neighbors_in, neighbors_out

def calc_msgs_actual(msgs, n_msg_iter, neighbors_in, neighbors_out):
    n_agents = msgs.shape[1]
    n_neighbors_in = neighbors_in.sum(axis=1)
    n_neighbors_out = neighbors_out.sum(axis=1)
    n_msgs_sent = np.zeros_like(msgs) # calculated as number of sent msgs
    n_msgs_recv = np.zeros_like(msgs) # calculated as number of received msgs
    n_msgs_sent_possible = (n_neighbors_out * n_msg_iter)[None, :]
    n_msgs_recv_possible = (n_neighbors_in * n_msg_iter)[None, :]
    agents_cansend = np.ones(n_agents, dtype=bool)
    agents_canrecv = np.ones(n_agents, dtype=bool)
    for i, m in enumerate(msgs):
        agents_cansend[:] = True
        agents_canrecv[:] = True
        for iteration in range(n_msg_iter + 1):
            mask = m == iteration
            agents_canrecv[mask] = False
            n_msgs_sent[i] += agents_cansend * (neighbors_out & agents_canrecv[None, :]).sum(axis=1)
            n_msgs_recv[i] += agents_canrecv * (neighbors_in & agents_cansend[None, :]).sum(axis=1)
            agents_cansend[mask] = False
    return n_msgs_sent, n_msgs_recv, n_msgs_sent_possible, n_msgs_recv_possible



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Read two files via named parameters')
    parser.add_argument('-i', '--input', default='plot_advanced_input.json', help='Path to json file with experiment paths')
    parser.add_argument('-o', '--out', default='exps', help='File name prefix for output images')
    args = parser.parse_args()
    paths_json_filepath = args.input
    with open(paths_json_filepath, 'r', encoding='utf-8') as f:
        paths_json_data = json.load(f)
    home_dir = ""
    exp_dirs_names = paths_json_data["exp_dirs_names"]
    exp_dir_nomem = paths_json_data["exp_dir_nomemory"]
    opt_path = paths_json_data["flows_solver_ecmp"]
    opt_path_resolve = str(Path(opt_path).resolve())
    exp_dirs, exp_names = zip(*exp_dirs_names)
    exp_dirs = list(exp_dirs)
    exp_names = list(exp_names)
    out_filename_prefix = args.out

    ecmp_solver_out_filename = "plot_values_ecmp+solver.npz"
    # solver_out_filename = "plot_values_solver.npz"
    ecmp_solver_pathsave_filename = "plot_values_ecmp+solver_path.json"
    ecmp_solver_cached = False
    if all([os.path.isfile(filename) for filename in
        [ecmp_solver_out_filename, ecmp_solver_pathsave_filename]]):
        with open(ecmp_solver_pathsave_filename, 'r', encoding='utf-8') as f:
            opt_path_norm_data = json.load(f)
            opt_path_norm_check = opt_path_norm_data["ecmp+solver_path"]
            if os.path.normpath(opt_path_norm_check) == os.path.normpath(opt_path_resolve):
                ecmp_solver_cached = True
            else:
                print(opt_path_norm_check), os.path.normpath(opt_path_norm_check)
                print(opt_path_resolve), os.path.normpath(opt_path_resolve)
                print("cached path to solver+ecmp values doesn't match, will load ecmp and solver values from original individual files")
    if not ecmp_solver_cached:
        print(f"loading ecmp values from {opt_path}...")
        phi_ecmp_full = get_ecmp_value(opt_path, full=True)
        print(f"loading solver values from {opt_path}...")
        phi_opt_full = get_opt_value(opt_path, full=True)
        print("ecmp and solver values loaded successfully")
        np.savez(ecmp_solver_out_filename, ecmp=phi_ecmp_full, solver=phi_opt_full)
        with open(ecmp_solver_pathsave_filename, 'w', encoding='utf-8') as f:
            json.dump({"ecmp+solver_path": opt_path_resolve}, f, ensure_ascii=False)
        print(f"saved ecmp and solver values in faster accessible form into {ecmp_solver_out_filename}, "
              f"{ecmp_solver_pathsave_filename}, next time they will load faster "
              f"(until different path to original values is specified)")
    else:
        print("loading cached ecmp and solver values...")
        phi_ecmp_opt_data = np.load(ecmp_solver_out_filename)
        phi_ecmp_full = phi_ecmp_opt_data['ecmp']
        phi_opt_full = phi_ecmp_opt_data['solver']
        print("ecmp and solver values loaded successfully")

    # These are for Abilene topology. for different topology, print these arrays in maroh code
    # (search where incoming_links are created) and paste here.
    # And change n_msg_iter to message_iterations from SAMAROH configs for different topology if needed.
    incoming_links = np.array([0, 0, 1, 1, 2, 2, 3, 3, 3, 4, 5, 5, 6, 6, 6, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12, 12, 12, 13, 13, 14, 15, 15, 16, 16, 17, 18, 18, 18, 19, 19])
    outcoming_links = np.array([18, 19, 15, 16, 5, 6, 8, 9, 10, 3, 12, 13, 7, 9, 10, 2, 4, 5, 11, 13, 17, 19, 4, 6, 7, 8, 10, 14, 16, 0, 11, 12, 17, 18, 1, 7, 8, 9, 14, 15])
    # incoming_links2 = np.array([0, 1, 2, 3, 4, 4, 5, 5, 6, 6, 7, 7, 7, 8, 8, 8, 9, 9, 9, 10, 10, 10, 11, 11, 12, 12, 13, 13, 14, 14, 15, 15, 16, 16, 17, 17, 18, 18, 19, 19])
    # outcoming_links2 = np.array([1, 0, 3, 2, 5, 6, 4, 6, 4, 5, 8, 9, 10, 7, 9, 10, 7, 8, 10, 7, 8, 9, 12, 13, 11, 13, 11, 12, 15, 16, 14, 16, 14, 15, 18, 19, 17, 19, 17, 18])
    n_msg_iter = 3
    neighbors_in, neighbors_out = get_neighbors(incoming_links, outcoming_links)

    # n_episodes_use = 2000
    n_episodes_use = len(phi_ecmp_full)

    phi_ecmp = np.copy(phi_ecmp_full)[:n_episodes_use]
    phi_opt = np.copy(phi_opt_full)[:n_episodes_use]
    path_nomem = os.path.join(home_dir, exp_dir_nomem)
    _, values_nomem, _, _, _, _ = load_data([path_nomem], ["phi_values"])
    phi_nomem = np.array(values_nomem[0]["phi_values"])[-len(phi_opt_full):][:n_episodes_use] # evaluation on test data is in this experiment's last episodes
    phi_arrays = []
    phi_comparison_arrays = []
    phi_comparisons_nomem = (phi_nomem - phi_opt) / (phi_ecmp - phi_opt)
    acc_table_rows = []
    # classes = np.unique(actions_gr_true.flatten())
    classes = np.array([0, 1])
    classifier_data_sizes = {}
    for exp_dir, exp_name in zip(exp_dirs, exp_names):
        _, values, _, _, _, _ = load_data([exp_dir], ["phi_values", "messages_agents", "actions_gr", "actions_gr_true"])
        phi = np.array(values[0]["phi_values"])[:n_episodes_use]
        msgs = np.array(values[0]["messages_agents"])[:n_episodes_use]
        actions_gr = np.array(values[0]["actions_gr"])[:n_episodes_use]
        actions_gr_true = np.array(values[0]["actions_gr_true"])[:n_episodes_use]
        n_episodes = msgs.shape[0]
        n_horizons = msgs.shape[1]
        n_agents = msgs.shape[-1]

        msgs = msgs.reshape(n_episodes * n_horizons, *msgs.shape[2:])
        n_msgs_sent, n_msgs_recv, n_msgs_sent_possible, n_msgs_recv_possible = calc_msgs_actual(msgs, n_msg_iter, neighbors_in, neighbors_out)
        actions_gr = actions_gr.reshape(n_episodes * n_horizons, *actions_gr.shape[2:])
        actions_gr_true = actions_gr_true.reshape(n_episodes * n_horizons, *actions_gr_true.shape[2:])
        # n_msg_iter = msgs.max()

        accuracy_stats, coverage_stats, accuracy, coverage = calc_accuracy(msgs, actions_gr, actions_gr_true, n_msg_iter)

        phi_comparisons = (phi - phi_opt) / (phi_ecmp - phi_opt)
        phi_arrays.append(phi)
        phi_comparison_arrays.append(phi_comparisons)
        print()
        print(f"{exp_name}:")
        msg_frac = msgs.mean() / n_msg_iter # fraction of exchange iterations done
        msg_frac_actual = n_msgs_sent.sum(axis=1).mean() / n_msgs_sent_possible.sum(axis=1)[0] # fraction of exchanges done
        msg_frac_actual2 = n_msgs_recv.sum(axis=1).mean() / n_msgs_recv_possible.sum(axis=1)[0] # fraction of exchanges done
        assert msg_frac_actual == msg_frac_actual2

        n_reprs_mean = np.nan
        n_reprs_std = np.nan
        try:
            memory_type = get_config_value(exp_dir, "memory_type")
            if memory_type is not None:
                memory_type = memory_type.lower().replace("'", "").replace('"', '')
                if memory_type == "epsilon_net":
                    reprs_path = get_config_value(exp_dir, "representatives_path")
                    if reprs_path is not None:
                        reprs_path = reprs_path.strip()
                        while reprs_path[0] == "'" or reprs_path[0] == '"':
                            reprs_path = reprs_path[1:-1]
                        lengths = []
                        for agent in range(n_agents):
                            for i in range(n_msg_iter):
                                reprs = np.load(os.path.join(reprs_path, f"reprs_a{agent:02}_i{i}.npz"))
                                lengths.append(len(reprs['centers']))
                        n_reprs_mean = np.mean(lengths)
                        n_reprs_std = np.std(lengths, ddof=1)
                elif memory_type == "knn_classifier":
                    reprs_path = get_config_value(exp_dir, "representatives_path")
                    if reprs_path is not None:
                        reprs_path = reprs_path.strip()
                        while reprs_path[0] == "'" or reprs_path[0] == '"':
                            reprs_path = reprs_path[1:-1]
                        if reprs_path not in classifier_data_sizes:
                            _, train_values, _, _, _, _ = load_data([reprs_path], ["actions_gr"])
                            train_actions = np.array(train_values[0]["actions_gr"])
                            n_reprs_mean = train_actions.shape[0] * train_actions.shape[1]
                            classifier_data_sizes[reprs_path] = n_reprs_mean
                        else:
                            n_reprs_mean = classifier_data_sizes[reprs_path]
                        n_reprs_std = 0
                elif memory_type == "centroids":
                    n_reprs_mean = len(classes)
                    n_reprs_std = 0
                elif memory_type == "cluster":
                    memory_size = get_config_value(exp_dir, "memory_size")
                    if memory_size is not None:
                        n_reprs_mean = int(memory_size)
                        n_reprs_std = 0
        except Exception as e:
            pass

        print(f"fraction of exchange iterations done: {msg_frac:.3f}")
        print(f"fraction of actual exchanges done: {msg_frac_actual:.3f}")
        print(f"horizon lengths distribution: {Counter(msgs.max(axis=1))}")
        print(f"mean phi (no memory):          {phi_nomem.mean():.3f} ± {phi_nomem.std(ddof=1):.3f}")
        print(f"mean phi ({exp_name}): {phi.mean():.3f} ± {phi.std(ddof=1):.3f}")
        print(f"mean phi (solver):             {phi_opt.mean():.3f} ± {phi_opt.std(ddof=1):.3f}")
        print(f"mean phi (ECMP):               {phi_ecmp.mean():.3f} ± {phi_ecmp.std(ddof=1):.3f}")
        print(f"mean phi position between ECMP and solver (no memory):          {phi_comparisons_nomem.mean():.3f} ± {phi_comparisons_nomem.std(ddof=1):.3f}")
        print(f"mean phi position between ECMP and solver ({exp_name}): {phi_comparisons.mean():.3f} ± {phi_comparisons.std(ddof=1):.3f}")
        print(f"all iterations: mean agent accuracy = {accuracy:.4f}, mean agent coverage = {coverage:.4f}")
        for i in range(n_msg_iter):
            print(f"iteration {i}: mean agent accuracy = {accuracy_stats[i]:.4f}, mean agent coverage = {coverage_stats[i]:.4f}")
        print(f"fraction of ReadOut NN usage = {coverage_stats[n_msg_iter]:.4f}")
        print(f"mean n_representatives = {n_reprs_mean:.0f} ± {n_reprs_std:.0f}")
        acc_table_rows.append([exp_name, msg_frac, msg_frac_actual, accuracy, coverage] +
            [accuracy_stats[i] for i in range(n_msg_iter)] +
            [coverage_stats[i] for i in range(n_msg_iter)] +
            [n_reprs_mean, n_reprs_std])

    columns = [
        "classifier",
        "fraction of exchange iterations done",
        "fraction of exchanges done",
        "accuracy",
        "coverage",
    ] + [f"accuracy (it.{i})" for i in range(n_msg_iter)] + [f"coverage (it.{i})" for i in range(n_msg_iter)] + [
        "representatives (mean)",
        "representatives (std)",
    ]

    df = pd.DataFrame(acc_table_rows, columns=columns)
    out_filename_table = f"{out_filename_prefix}_table.csv"
    df.to_csv(out_filename_table)
    print(f"saved table to {out_filename_table}")

    plt.figure(figsize=(10,5), dpi=300)
    bp = plt.boxplot([phi_opt, phi_nomem] + phi_arrays + [phi_ecmp], tick_labels=["solver", "no memory"] + exp_names + ["ECMP"],
                    flierprops=dict(markersize=2, markeredgecolor=None, marker='o', markeredgewidth=0.5, markerfacecolor='white'))
    medians = [bp['medians'][i].get_ydata()[0] for i in range(len(bp['medians']))]
    for i, median in enumerate(medians, start=1):
        plt.text(i, median, f'{median:.3f}',
                 ha='center', va='bottom',
                 bbox=dict(facecolor='white', edgecolor='none', alpha=0.5))
    plt.xticks(rotation=45, ha='right')
    plt.ylabel("phi")
    plt.grid()
    out_filename_plot1 = f"{out_filename_prefix}_plot_phi"
    plt.savefig(out_filename_plot1, bbox_inches='tight')
    print(f"saved phi plot to {out_filename_plot1}.png")
    plt.close()

    plt.figure(figsize=(10,5), dpi=300)
    bp = plt.boxplot([phi_comparisons_nomem] + phi_comparison_arrays, tick_labels=["no memory"] + exp_names,
                    flierprops=dict(markersize=2, markeredgecolor=None, marker='o', markeredgewidth=0.5, markerfacecolor='white'))
    medians = [bp['medians'][i].get_ydata()[0] for i in range(len(bp['medians']))]
    for i, median in enumerate(medians, start=1):
        plt.text(i, median, f'{median:.3f}',
                 ha='right', va='bottom',
                bbox=dict(facecolor='white', edgecolor='none', alpha=0.7))
    plt.xticks(rotation=45, ha='right')
    plt.ylabel("phi position (solver—ECMP)")
    plt.ylim(0, 2)
    plt.grid()
    out_filename_plot2 = f"{out_filename_prefix}_plot_phi_normalized"
    plt.savefig(out_filename_plot2, bbox_inches='tight')
    print(f"saved phi plot to {out_filename_plot2}.png")
    plt.close()
