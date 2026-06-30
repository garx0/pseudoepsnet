import numpy as np
from sklearn.cluster import MiniBatchKMeans, KMeans, AgglomerativeClustering
from random import random
import time
import logging
import os
import re
from collections import defaultdict

from dte_stand.algorithm.mate.lib.classifiers import ClassifierFactory, MetricSpace
from types import SimpleNamespace

ACT_VEC_UPDATE = 0
ACT_VEC_NEW_UPDATE = 1
ACT_VEC_USE = 2
ACT_VEC_IGNORE = 3
ACT_VEC_USE_AND_UPDATE = 4

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

def load_maroh_data(path):
    '''
    path: directory containing states_*-*.npz file(s) and action_gr_*-*.npz file(s)
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
    actions = np.array(values['actions_gr'])
    states = {}
    for key in values.keys():
        if key != 'actions_gr':
            states[int(key)] = np.array(values[key])
    n_iterations = len(states.keys()) - 1
    n_episodes = actions.shape[0]
    n_horizons = actions.shape[1]
    n_agents = actions.shape[-1]
    for k, v in states.items():
        states[k] = v.reshape(n_episodes * n_horizons, n_agents, v.shape[-1])
    actions = actions.reshape(n_episodes * n_horizons, n_agents)
    return states, actions

class StatesMemory(object):
    def __init__(self, iteration_id, max_size, max_new_size, threshold, n_agents, state_size, n_actions, P, var, initial_state=None):
        self.iteration_id = iteration_id
        self.mem_vec = np.empty([0, n_agents, state_size])
        self.act_vec = np.empty([0, n_agents, 16])
        self.mem_vec_new = np.empty([0, n_agents, state_size])
        self.act_vec_new = np.empty([0, n_agents, 16]) # n_actions

        self.max_size = max_size # Better to set 2**n
        self.max_new_size = max_new_size
        self.n_agents = n_agents
        self.n_actions = n_actions
        self.state_size = state_size

        self.mem_vec_full = None
        self.act_vec_full = None
        if initial_state is not None:
            self.mem_vec_full = initial_state[:, :, :state_size]
            self.act_vec_full = initial_state[:, :, state_size:-1]
            self.mem_vec = np.copy(self.mem_vec_full)
            self.act_vec = np.copy(self.act_vec_full)

        # self.cur_size = 0
        self.clustered = False
        self.threshold = threshold # TODO: This should be less than minimal cluster distance
        self.experience_gamma = 1.00 # Probability of using mem_vec

        self.P = P
        self.var = var

    # def topology_changes(self, inserts, removes, n_agents):
    #     self.mem_vec = np.insert(self.mem_vec, inserts, np.zeros(self.state_size), axis=1)
    #     self.mem_vec_new = np.insert(self.mem_vec_new, inserts, np.zeros(self.state_size), axis=1)
    #     self.act_vec = np.insert(self.act_vec, inserts, np.zeros(16), axis=1)
    #     self.act_vec_new = np.insert(self.act_vec_new, inserts, np.zeros(16), axis=1)

    #     self.mem_vec = np.delete(self.mem_vec, removes, axis=1)
    #     self.mem_vec_new = np.delete(self.mem_vec_new, removes, axis=1)
    #     self.act_vec = np.delete(self.act_vec, removes, axis=1)
    #     self.act_vec_new = np.delete(self.act_vec_new, removes, axis=1)

    #     self.n_agents = n_agents

    def topology_changes(self, indices):
        if self.mem_vec_full is None: # training mode
            return
        self.mem_vec = self.mem_vec_full[:, indices, :]
        self.act_vec = self.act_vec_full[:, indices, :]

        self.mem_vec_new = self.mem_vec_full[:0, indices, :]
        self.act_vec_new = self.act_vec_full[:0, indices, :]

        self.n_agents = len(indices)


    def update_memory(self, new_states):
        # returns code and action
        ...

    def update_action(self, new_acts):
        ...

    def update_action_new(self, new_acts):
        ...

class EpsilonNetMemory(StatesMemory):
    def __init__(self, iteration_id, max_size, threshold, n_agents, state_size, n_actions, P, var, initial_state=None,
                 representatives_path='representatives'):
        super().__init__(iteration_id, max_size, max_size // 2, threshold, n_agents, state_size, n_actions, P, var, initial_state)
        self.representatives_path = representatives_path
        args = SimpleNamespace(delta=0, theta=0, buffer_size=100000, dist_percentile=0.1,
                               use_acceleration=True, verbose=True, searcher="balltree")
        self.clfs = []
        if not os.path.isfile(f"{self.representatives_path}/reprs_a-1_i{self.iteration_id}.npz"):
            for agent in range(n_agents):
                clf = ClassifierFactory.create(
                                "epsilon_net",
                                metric = MetricSpace(metric="euclidean"),
                                agent = agent,
                                iteration = iteration_id,
                                config = args)
                clf.load_representatives(f"{self.representatives_path}/reprs_a{agent:02}_i{self.iteration_id}.npz")
                self.clfs.append(clf)
        else:
            clf = ClassifierFactory.create(
                            "epsilon_net",
                            metric = MetricSpace(metric="euclidean"),
                            agent = -1,
                            iteration = iteration_id,
                            config = args)
            clf.load_representatives(f"{self.representatives_path}/reprs_a-1_i{self.iteration_id}.npz")
            for agent in range(n_agents):
                self.clfs.append(clf)

    def update_memory(self, new_states, update=True):
        if self.experience_gamma == 0.0:
            return ACT_VEC_IGNORE, None

        actions = []
        are_confident = []
        codes = []
        for agent, clf in enumerate(self.clfs):
            action, _, is_confident, _, _, _ = clf.predict_one(new_states[agent])
            if not is_confident:
                action = -1
            actions.append(action)
            are_confident.append(is_confident)
            codes.append(ACT_VEC_USE if is_confident else ACT_VEC_IGNORE)
        return codes, actions

    def update_action(self, new_acts):
        pass

    def update_action_new(self, new_acts):
        pass

class KNNClassifierMemory(StatesMemory):
    def __init__(self, iteration_id, max_size, threshold, n_agents, state_size, n_actions, P, var, initial_state=None,
                first_iter_active=0, all_agents=False, representatives_path='classifier_data'):
        super().__init__(iteration_id, max_size, max_size // 2, threshold, n_agents, state_size, n_actions, P, var, initial_state)
        self.first_iter_active = first_iter_active # don't use memory on message iterations before this one
        self.all_agents = all_agents
        self.representatives_path = representatives_path
        if self.iteration_id >= self.first_iter_active:
            self.clfs = []
            states, actions = load_maroh_data(self.representatives_path)
            if not self.all_agents:
                for agent in range(n_agents):
                    clf = ClassifierFactory.create(
                                    "kNN",
                                    metric = MetricSpace(metric="euclidean"),
                                    agent = agent,
                                    iteration = iteration_id)
                    X = states[iteration_id][:, agent]
                    y = actions[:, agent]
                    clf.fit(X, y)
                    self.clfs.append(clf)
            else:
                clf = ClassifierFactory.create(
                                "kNN",
                                metric = MetricSpace(metric="euclidean"),
                                agent = -1,
                                iteration = iteration_id)
                X = states[iteration_id]
                X = X.reshape(X.shape[0] * X.shape[1], X.shape[2])
                y = actions
                y = y.reshape(y.shape[0] * y.shape[1])
                clf.fit(X, y)
                for agent in range(n_agents):
                    self.clfs.append(clf)

    def update_memory(self, new_states, update=True):
        if self.experience_gamma == 0.0 or self.iteration_id < self.first_iter_active:
            return [ACT_VEC_IGNORE for _ in new_states], [-1 for _ in new_states]
        actions = []
        # are_confident = []
        codes = []
        for agent, clf in enumerate(self.clfs):
            action, _, is_confident, _, _, _ = clf.predict_one(new_states[agent])
            if not is_confident:
                action = -1
            actions.append(action)
            # are_confident.append(is_confident)
            codes.append(ACT_VEC_USE if is_confident else ACT_VEC_IGNORE)
        return codes, actions

    def update_action(self, new_acts):
        pass

    def update_action_new(self, new_acts):
        pass

class CentroidsMemory(StatesMemory):
    def __init__(self, iteration_id, max_size, threshold, n_agents, state_size, n_actions, P, var, initial_state=None,
                first_iter_active=0, all_agents=False, representatives_path='classifier_data'):
        super().__init__(iteration_id, max_size, max_size // 2, threshold, n_agents, state_size, n_actions, P, var, initial_state)
        self.first_iter_active = first_iter_active # don't use memory on message iterations before this one
        self.all_agents = all_agents
        self.representatives_path = representatives_path
        if self.iteration_id >= self.first_iter_active:
            self.clfs = []
            args = SimpleNamespace(theta=0, verbose=True, searcher="balltree")
            states, actions = load_maroh_data(self.representatives_path)
            if not self.all_agents:
                for agent in range(n_agents):
                    clf = ClassifierFactory.create(
                                    "centroids",
                                    metric = MetricSpace(metric="euclidean"),
                                    agent = agent,
                                    iteration = iteration_id,
                                    config = args)
                    X = states[iteration_id][:, agent]
                    y = actions[:, agent]
                    clf.fit(X, y)
                    self.clfs.append(clf)
            else:
                clf = ClassifierFactory.create(
                                "centroids",
                                metric = MetricSpace(metric="euclidean"),
                                agent = -1,
                                iteration = iteration_id,
                                config = args)
                X = states[iteration_id]
                X = X.reshape(X.shape[0] * X.shape[1], X.shape[2])
                y = actions
                y = y.reshape(y.shape[0] * y.shape[1])
                clf.fit(X, y)
                for agent in range(n_agents):
                    self.clfs.append(clf)

    def update_memory(self, new_states, update=True):
        if self.experience_gamma == 0.0 or self.iteration_id < self.first_iter_active:
            return [ACT_VEC_IGNORE for _ in new_states], [-1 for _ in new_states]
        actions = []
        # are_confident = []
        codes = []
        for agent, clf in enumerate(self.clfs):
            action, _, is_confident, _, _, _ = clf.predict_one(new_states[agent])
            if not is_confident:
                action = -1
            actions.append(action)
            # are_confident.append(is_confident)
            codes.append(ACT_VEC_USE if is_confident else ACT_VEC_IGNORE)
        return codes, actions

    def update_action(self, new_acts):
        pass

    def update_action_new(self, new_acts):
        pass
