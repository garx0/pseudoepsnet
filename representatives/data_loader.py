import numpy as np
import os
import re
from collections import defaultdict
from typing import List, Tuple

class MarohDataParser:
    def __init__(self, path, parameter_name):
        self._folder_path = path
        self._results = None
        self._max_episode = 0
        self._parameter_name = parameter_name
        self._parse_folder()

    def _parse_file(self, file_obj, start):
        data_list = np.load(file_obj)
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
                    continue
                if not file.is_file():
                    continue
                match = re.search(f'({self._parameter_name})_([0-9]+)-([0-9]+)\.np[y|z]', file.name)
                if not match:
                    continue
                with open(file.path, 'rb') as f:
                    self._parse_file(f, start=int(match[2]))
                    files_parsed += 1

def load_maroh_data(path, keep_episode: bool = True):
    '''
    result_path: directory containing states_*-*.npz file(s) and action_gr_*-*.npz file(s)
    '''
    values = {}
    params = ["states", "actions_gr"]
    for param in params:
        folder = MarohDataParser(path, parameter_name=param)
        if folder._results is None:
            raise Exception(f"couldn't read states_*-*.npz file(s) and action_gr_*-*.npz file(s) from {path}")
        for subparam, res_subparam in folder._results.items():
            max_episode = len(res_subparam)
            values[subparam] = [res_subparam[episode] for episode in range(0, max_episode)]
    if "actions_gr" in values:
        actions_key = "actions_gr"
    elif "actions_gr_true" in values:
        actions_key = "actions_gr_true"
    actions = np.array(values[actions_key])
    states = {}
    for key in values.keys():
        if key != actions_key:
            states[int(key)] = np.array(values[key])
    n_iterations = len(states.keys()) - 1
    n_episodes = actions.shape[0]
    n_horizons = actions.shape[1]
    n_agents = actions.shape[-1]
    print("N agents ", n_agents)
    print("N iterations ", n_iterations)
    link_state_size = states[1].shape[-1]
    link_state_size_0 = states[0].shape[-1]

    if not keep_episode:
        # convert to flat format (losing info about episode and horizon)
        dtype = np.dtype([
            ('agent', np.int32),
            ('state', np.float64, link_state_size),
            ('iteration', np.int32),
            ('action', np.int32)
        ])
    else:
        # convert to flat format (losing info about horizon)
        dtype = np.dtype([
            ('agent', np.int32),
            ('state', np.float64, link_state_size),
            ('iteration', np.int32),
            ('action', np.int32),
            ('episode', np.int32),
        ])

    data = np.empty(n_episodes * n_horizons * n_agents * (n_iterations + 1), dtype=dtype)
    i = 0
    for episode_idx in range(n_episodes):
        for horizon_idx in range(n_horizons):
            for agent_idx in range(n_agents):
                for iteration in range(0, n_iterations + 1):
                    data[i]['agent'] = agent_idx

                    # len(state) == link_state_size if iteration > 0 else link_state_size_0
                    state = states[iteration][episode_idx, horizon_idx, agent_idx]
                    data[i]['state'][:] = 0
                    data[i]['state'][:len(state)] = state # padded with zeros if iteration = 0

                    data[i]['iteration'] = iteration
                    data[i]['action'] = actions[episode_idx, horizon_idx, agent_idx]
                    if keep_episode:
                        data[i]['episode'] = episode_idx
                    i += 1
    return data


class DataLoader:
    """Data loader from file"""

    @staticmethod
    def load_sequence_data(path: str, keep_episode: bool = True) -> np.ndarray:
        """Load data from file or directory.
        If path is a directory, assumes data is in MAROH output format, and
        converts it to the required format on the fly."""
        if os.path.isdir(path):
            data = load_maroh_data(path, keep_episode=keep_episode)
        else:
            data = np.load(path, allow_pickle=True)
        print(f"{len(data)} states and actions loaded in total from {path}")
        return data

    @staticmethod
    def prepare_training_data(data: np.ndarray, agent_id: int, iteration_id: int,
                              train_ratio: float = 0.6, random_seed: int = 42,
                              data_test: np.ndarray = None) ->  \
                              Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Prepare training and test data for a specific agent and iteration

        Args:
            data: Loaded data
            agent_id: Agent ID for filtering
            iteration_id: Iteration ID for filtering
            train_ratio: Fraction of data for training from data (if data_test not specified)
            data_test: Test data (if None, data is split into train and test)
        Returns:
            (training_points, test_points, training_labels, test_labels)
        """
        # Filter data by agent and iteration
        if agent_id >= 0:
            if iteration_id >= 0:
                data2 = data[(data['agent'] == agent_id) & (data['iteration'] == iteration_id)]
            else:
                data2 = data[data['agent'] == agent_id]
        elif iteration_id >= 0:
            data2 = data[data['iteration'] == iteration_id]
        else:
            data2 = data[:]
        X = data2['state']
        y = data2['action']

        if len(X) == 0:
            raise ValueError(f"No data for agent {agent_id}, iteration {iteration_id}")

        shuffle_before_split = True
        shuffle_after_split = True

        if data_test is not None:
            if agent_id >= 0:
                if iteration_id >= 0:
                    data_test2 = data_test[(data_test['agent'] == agent_id) & (data_test['iteration'] == iteration_id)]
                else:
                    print("iteration == -1, selecting all iterations")
                    data_test2 = data_test[data_test['agent'] == agent_id]
            elif iteration_id >= 0:
                print("agent == -1, selecting all agents")
                data_test2 = data_test[data_test['iteration'] == iteration_id]
            else:
                print("agent == -1, iteration == -1, selecting all agents and iterations")
                data_test2 = data_test[:]
            X_test = data_test2['state']
            y_test = data_test2['action']
            X_train = X
            y_train = y
        elif 'episode' in data2.dtype.fields:
            X_episode = data2['episode']
            episodes = np.unique(X_episode)
            if shuffle_before_split:
                np.random.seed(random_seed)
                np.random.shuffle(episodes)
                shuffle_idx = np.zeros(len(X), dtype=int)
                shift = 0
                for episode in episodes:
                    episode_indices = np.where(X_episode == episode)[0]
                    shuffle_idx[shift : shift + len(episode_indices)] = episode_indices
                    shift += len(episode_indices)
                X = X[shuffle_idx]
                y = y[shuffle_idx]
                X_episode = X_episode[shuffle_idx]

            # Split into training and test sets
            train_size_init = int(len(X) * train_ratio)
            # Make sure each episode goes entirely into one of the two sets
            wh = np.where(X_episode == X_episode[train_size_init - 1])[0]
            train_size = wh[-1] + 1
            if train_size >= len(X):
                train_size = wh[0]
            X_train = X[:train_size]
            X_test = X[train_size:]
            y_train = y[:train_size]
            y_test = y[train_size:]
        else:
            if shuffle_before_split:
                shuffle_idx = np.arange(len(X))
                np.random.seed(random_seed)
                np.random.shuffle(shuffle_idx)
                X = X[shuffle_idx]
                y = y[shuffle_idx]

            # Split into training and test sets
            train_size = int(len(X) * train_ratio)
            X_train = X[:train_size]
            X_test = X[train_size:]
            y_train = y[:train_size]
            y_test = y[train_size:]

        if shuffle_after_split:
            np.random.seed(random_seed)
            shuffle_idx2 = np.arange(len(X_train))
            np.random.shuffle(shuffle_idx2)
            X_train = X_train[shuffle_idx2]
            y_train = y_train[shuffle_idx2]

        print(f"Data prepared: {len(X_train)} training, {len(X_test)} test samples")

        return X_train, X_test, y_train, y_test

    @staticmethod
    def prepare_stream_data(data: np.ndarray, agent_id: int, iteration_id: int) -> List[np.ndarray]:
        """Prepare streaming data (without labels)"""
        stream_data = []
        for item in data:
            if item['agent'] == agent_id and item['iteration'] == iteration_id:
                stream_data.append(item['state'])
        return stream_data
