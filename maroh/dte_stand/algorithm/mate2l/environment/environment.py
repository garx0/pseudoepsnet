# import gin.tf
import os
import copy
import math
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import itertools
import sys
sys.path.append('./dte_stand')

from collections import defaultdict
from dte_stand.data_structures import HashWeights, Flow, InputData
from dte_stand.algorithm.mate2l.config import MateActions
from dte_stand.history import HistoryTracker
from networkx.drawing.nx_agraph import write_dot
from typing import Optional, Iterable

import warnings
warnings.simplefilter("error", category=RuntimeWarning)

DEFAULT_EDGE_ATTRIBUTES = {
    'increments': 1,
    'reductions': 1,
    'weight': 0.0,
}


class WelfordStateNormalizer:
    def __init__(self, size: int):
        self.size = size
        self.reset()

    def reset(self):
        self._state_num = 1
        self._state_mean = np.zeros(shape=self.size, dtype=np.float64)
        self._state_mean_diff = np.ones(shape=self.size, dtype=np.float64)

    def __call__(self, state):
        if self._state_num == 1:
            self._state_num += 1
            return state
        state_old = self._state_mean
        self._state_mean += (state - state_old) / self._state_num
        self._state_mean_diff += (state - state_old) * (state - state_old)
        self._state_num += 1
        return (state - self._state_mean) / np.sqrt(self._state_mean_diff / self._state_num)


# @gin.configurable
class Environment(object):
    def __init__(self,
                 current_topology,
                 path_calculator,
                 hash_function,
                 action_config: MateActions,
                 phi_func,
                 env_type='Test',
                 traffic_profile='gravity_1',
                 routing='dxhash',
                 init_sample=0,
                 seed_init_weights=1,
                 weights_initialization='equal',
                 min_weight=1.0,
                 max_weight=7.0,
                 weight_change=1.0,
                 weight_update='sum',
                 weigths_to_states=True,
                 link_traffic_to_states=True,
                 probs_to_states=False,
                 reward_magnitude='link_traffic',
                 # base_reward='min_max',
                 base_reward='phi',
                 reward_computation='change',
                 # reward_computation='value',
                 base_dir='topologies',
                 graph_dir='dte_stand/algorithm/mate/graphs',
                 base_data_dir='data_examples',
                 topology='test.gml',
                 current_flows=[],
                 clip_state = False
                 ):

        env_type = [env for env in env_type.split('+')]
        self.env_type = env_type
        self.traffic_profile = traffic_profile
        self.routing = routing
        self.topology = topology
        self.base_data_dir = base_data_dir

        self.eval_mode = False

        self.clip_state = clip_state

        self.weights_initialization = weights_initialization

        self.G = None

        self.path_calculator = path_calculator

        self.num_sample = init_sample - 1
        self.seed_init_weights = seed_init_weights
        self.min_weight = min_weight
        self.max_weight = max_weight
        self.weight_change = weight_change
        self.weight_update = weight_update

        num_features = 0
        self.weigths_to_states = weigths_to_states
        if self.weigths_to_states:
            num_features += 1
        self.link_traffic_to_states = link_traffic_to_states
        if self.link_traffic_to_states:
            num_features += 1
        self.probs_to_states = probs_to_states
        if self.probs_to_states:
            num_features += 2
        self.num_features = num_features
        self.reward_magnitude = reward_magnitude
        self.base_reward = base_reward
        self.reward_computation = reward_computation
        self.base_dir = base_dir
        self.graph_dir = graph_dir
        if current_topology is not None:
            self.current_topology = current_topology
            self.initialize_environment()
            self.get_weights()

        self.tracker = HistoryTracker()
        self.phi_history = []
        self.phi = phi_func
        self.current_flows = current_flows
        self.hash_weights: Optional[HashWeights] = None
        self.hash_function = hash_function
        self.prev_edges = []
        self.act_types = {}
        if action_config.addition.action:
            self.act_types['+'] = (1, action_config.addition.value)
        if action_config.subtraction.action:
            self.act_types['-'] = (1, action_config.subtraction.value)
        if action_config.multiplication.action:
            self.act_types['*'] = (action_config.multiplication.value, 0)
        if action_config.multiplication2.action:
            self.act_types['*2'] = (action_config.multiplication2.value, 0)
        if action_config.division.action:
            self.act_types['/'] = (action_config.division.value, 0)
        if action_config.division2.action:
            self.act_types['/2'] = (action_config.division2.value, 0)
        if action_config.zero.action:
            self.act_types['0'] = (1, 0)

        self.state_normalizer = WelfordStateNormalizer(10 * 2)

    def get_current_flows(self, current_flows):
        self.current_flows = current_flows

    def set_current_topology(self, current_topo):
        self.current_topology = current_topo
        self.initialize_environment()
        self.get_weights()

    def _calculate_current_bandwidth(self, topology: nx.MultiDiGraph, flows: Iterable[Flow],
                                     hash_weights: HashWeights, link=None, horizon=None, num_sample=None) -> None:
        if hash_weights is None:
            return
        self.hash_function.run(topology, flows, hash_weights, link, use_flow_memory=False)

        self._get_link_traffic()

    def load_topology_object(self, current_topology=None):
        # load topology to self.topology_object
        try:
            if current_topology:
                self.topology_object = current_topology
            else:
                nx_file = os.path.join(self.base_dir, self.topology)
                self.topology_object = nx.MultiDiGraph(nx.read_gml(nx_file, destringizer=int))

        except:
            print("Bad input of topology.gml")

    def update_topology(self, topology: nx.MultiDiGraph) -> None:
        self.topology_object = topology
        self.generate_graph()
        self.get_weights()
        # self.tracker.reset()

    def initialize_environment(self, num_sample=None, equal=False):
        if num_sample is not None:
            self.num_sample = num_sample
        else:
            self.num_sample += 1
        self.load_topology_object(self.current_topology)
        self.generate_graph(equal=equal)

    def next_sample(self):
        if len(self.env_type) > 1:
            self.initialize_environment()
        else:
            self.num_sample += 1
            self._reset_edge_attributes()

    def define_num_sample(self, num_sample):
        self.num_sample = num_sample - 1

    def end_iteration(self):
        self.tracker.add_value('phi_values', self.phi_history)
        self.tracker.end_iteration()
        self.phi_history.clear()
        self.hash_function.end_iteration()

    # def reset(self, topo = None, flows = None):
    #     if topo:
    #         self.topology_object = topo
    #         self.generate_graph()
    #     if flows:
    #         self.current_flows = flows
    #     self.get_weights()
    #     self.get_hash_weights()
    #     self._calculate_current_bandwidth(self.G, self.current_flows, self.hash_weights)
    #     self.reward_measure = self.compute_reward_measure()
    #     self.state_normalizer.reset()


    def reset(self, topo = None, flows = None, hash_weights = None, new_topo = False, equal=False):
        #print('reset env')
        if topo is not None or flows is not None:
            # new sample
            if new_topo == True:
                self.topology_object = topo
                self.current_topology = topo
                self.path_calculator.prepare_iteration(topo)
            #print('iteration prepared')
            self.current_flows = flows
            self.initialize_environment(equal=equal)
            # self.next_sample()
        else:
            if self.seed_init_weights is None:
                self._define_init_weights(equal=equal)
            self._reset_edge_attributes()
        if hash_weights is not None: self.hash_weights_to_weights(hash_weights)
        self.get_weights()
        self.get_hash_weights()
        self._calculate_current_bandwidth(self.G, self.current_flows, self.hash_weights)
        self.reward_measure = self.compute_reward_measure()
        self.initial_measure = self.reward_measure
        self.state_normalizer.reset()

        # self.set_target_measure()
        # return self.get_state()

    def hash_weights_to_weights(self, hash_weights: HashWeights):
        for i, j, m in self.G.edges:
            w = hash_weights.get_weight(i, j, j, m)
            if w is not None: self.G[i][j][m]['weight'] = w


    def generate_graph(self, hash_weights : HashWeights = None, equal=False):
        G = copy.deepcopy(self.topology_object)
        self.n_nodes = G.number_of_nodes()
        self.n_links = G.number_of_edges()
        self.weights = [0.0] * self.n_links
        self.link_traffic = [0.0] * self.n_links
        self._define_init_weights(equal=equal) # initialize n_links weights in self.init_weights
        idx = 0
        link_ids_dict = {}
        for i, j, m in G.edges: # i, j - node lables
            G[i][j][m]['label'] = G[i][j][m]['id']
            G[i][j][m]['id'] = idx
            G[i][j][m]['increments'] = 1
            G[i][j][m]['reductions'] = 1
            G[i][j][m]['weight'] = copy.deepcopy(self.init_weights[idx])
            link_ids_dict[idx] = (i, j, m)
            idx += 1
        self.G = G
        incoming_links, outcoming_links, incoming_links_2, outcoming_links_2, links_opposites = self._generate_link_indices_and_adjacencies()
        self.G.add_node('graph_data', link_ids_dict=link_ids_dict, incoming_links=incoming_links,
                        outcoming_links=outcoming_links, il2=incoming_links_2, ol2=outcoming_links_2, opposites=links_opposites)

    def get_weights(self):
        max_weight = self.max_weight * 3
        for i, j, m, edge_data in self.G.edges(keys=True, data=True):
            self.weights[edge_data['id']] = edge_data["weight"] / max_weight

    def get_state(self):
        state_features = []
        if self.link_traffic:
            if self.clip_state:
                state_features.append(np.clip(self.link_traffic, 0.0, 1.0))
            else:
                state_features.append(self.link_traffic)
        if self.weigths_to_states:
            state_features.append(self.weights)
        if self.probs_to_states:
            state_features.append(self.p_in)
            state_features.append(self.p_out)
        #return self.state_normalizer(np.fromiter(itertools.chain(*state_features), dtype=np.float32))
        return np.fromiter(itertools.chain(*state_features), dtype=np.float32)

    def update_weights(self, link, action_value, action_type, step_back=False):
        i, j, m = link
        if self.weight_update == 'min_max':
            if action_value == 0:
                self.G[i][j][m]['weight'] = max(
                    self.G[i][j][m]['weight'] - self.weight_change, self.min_weight)
            elif action_value == 1:
                self.G[i][j][m]['weight'] = min(
                    self.G[i][j][m]['weight'] + self.weight_change, self.max_weight)
        else:
            if self.weight_update == 'increment_reduction':
                if action_value == 0:
                    self.G[i][j][m]['reductions'] += 1
                elif action_value == 1:
                    self.G[i][j][m]['increments'] += 1
                self.G[i][j][m]['weight'] = self.G[i][j][m]['increments'] / \
                    self.G[i][j][m]['reductions']
            elif self.weight_update == 'sum':
                if step_back:
                    self.G[i][j][m]['weight'] -= self.weight_change
                else:
                    self.G[i][j][m]['weight'] += self.act_types[action_type][1]
                    self.G[i][j][m]['weight'] *= self.act_types[action_type][0]
            else:
                raise NotImplementedError

    def step(self, action, action_type, step_back=False, horizon=None, num_sample=None, last=False):
        link = self.G.nodes()['graph_data']['link_ids_dict'][action]
        # print(action, action_type, link)
        self.update_weights(link, 0, action_type, step_back) # self.G[i][j][m]['weight'] += 1
        self.get_weights() # self.weights (for state)
        self.get_hash_weights()
        self._calculate_current_bandwidth(self.G, self.current_flows, self.hash_weights, link, horizon, num_sample)
        state = self.get_state()
        reward = self._compute_reward(last=last)
        return state.astype(np.float32), reward

    def get_adj_matrix(self):
        if self.G == None: return None
        adj = np.diag(np.ones(self.n_links))
        links_in_node = defaultdict(list)
        for i, j, m in self.G.edges:
            links_in_node[j].append(self.G[i][j][m]['id'])
        for i, j, m in self.G.edges:
            for l in links_in_node[i]:
                adj[self.G[i][j][m]['id'], l] = 1
        return adj

    def multiple_step(self, action_types, step_back=False, horizon=None, num_sample=None, last=False):
        # print(self.act_types)
        # reward = []
        for link_n in range(len(action_types)):
            link = self.G.nodes()['graph_data']['link_ids_dict'][link_n]
            # print(link, action_types[link_n])
            self.update_weights(link, 0, action_types[link_n], step_back)
        self.get_weights()
        self.get_hash_weights()
        self._calculate_current_bandwidth(self.G, self.current_flows, self.hash_weights, None, horizon, num_sample)
        reward = self._compute_reward(last=last)
        state = self.get_state()
        # reward = self._compute_reward(last=last)
        # print("reward", reward)
        return state.astype(np.float32), reward

    def _define_init_weights(self, equal=False):
        # rng = np.random.default_rng(seed=self.seed_init_weights)
        # self.init_weights = rng.integers(self.min_weight, self.max_weight + 1, self.n_links)

        # add_coefs = [v[1] for v in self.act_types.values() if v[1] != 0]
        mul_coefs = [v[0] for v in self.act_types.values() if v[0] != 1]
        mul_coefs = np.sort(np.unique(mul_coefs))
        weights_are_exp = False
        if len(mul_coefs) == 1:
            weights_are_exp = True
        elif len(mul_coefs) == 2:
            mul_coefs_log = math.log(mul_coefs[1], mul_coefs[0])
            if abs(mul_coefs_log - round(mul_coefs_log)) < 1e-3:
                weights_are_exp = True
            assert weights_are_exp, f"{mul_coefs=}" # temporary condition. remove this check if going to use multiplicative actions with factors not being powers of each other
        if weights_are_exp:
            mul_coef = mul_coefs[0]
            # all actions do multiplicative weight changes with same factor mul_coef
            # then, we will always initialize the values as powers of mul_coef
            if equal == True or self.weights_initialization == 'equal':
                init_weights_base = np.full(self.n_links, 1)
            elif self.weights_initialization == 'deterministic':
                rng = np.random.default_rng(seed=self.seed_init_weights)
                init_weights_base = rng.integers(self.min_weight, self.max_weight + 1, self.n_links)
            elif self.weights_initialization == 'random':
                rng = np.random.default_rng(seed=None)
                init_weights_base = rng.integers(self.min_weight, self.max_weight + 1, self.n_links)
            elif self.weights_initialization == 'crandom':
                rng = np.random.default_rng(seed=None)
                init_weights_base = rng.uniform(self.min_weight, self.max_weight, self.n_links)
            else:
                raise ValueError(f"invalid weights_initialization value: {self.weights_initialization}")
            self.init_weights = mul_coef ** (init_weights_base - 1)
        else:
            if equal == True or self.weights_initialization == 'equal':
                self.init_weights = np.full(self.n_links, 1)
            elif self.weights_initialization == 'deterministic':
                rng = np.random.default_rng(seed=self.seed_init_weights)
                self.init_weights = rng.integers(self.min_weight, self.max_weight + 1, self.n_links)
            elif self.weights_initialization == 'random':
                rng = np.random.default_rng(seed=None)
                self.init_weights = rng.integers(self.min_weight, self.max_weight + 1, self.n_links)
            elif self.weights_initialization == 'crandom':
                rng = np.random.default_rng(seed=None)
                self.init_weights = rng.uniform(self.min_weight, self.max_weight, self.n_links)
            else:
                raise ValueError(f"invalid weights_initialization value: {self.weights_initialization}")
        # print(f"define_init_weights: {self.init_weights[:8]} ...")

    def _generate_link_indices_and_adjacencies(self):
        incoming_links = []
        outcoming_links = []
        incoming_links_2 = []
        outcoming_links_2 = []
        links_opposites = [0] * self.n_links
        for i in self.G.nodes():
            for j in self.G.neighbors(i):
                incoming_link_id = self.G[i][j][0]['id']
                links_opposites[self.G[i][j][0]['id']] = self.G[j][i][0]['id']
                for k in self.G.neighbors(j):
                    outcoming_link_id = self.G[j][k][0]['id']
                    if k != i:
                        incoming_links.append(incoming_link_id)
                        outcoming_links.append(outcoming_link_id)
        for i in self.G.nodes():
            neigh = [self.G[i][j][0]['id'] for j in self.G.neighbors(i)]
            for j in neigh:
                for k in neigh:
                    if k != j:
                        incoming_links_2.append(j)
                        outcoming_links_2.append(k)
        return incoming_links, outcoming_links, incoming_links_2, outcoming_links_2, links_opposites

    def _reset_edge_attributes(self, attributes=None):
        if attributes is None:
            attributes = list(DEFAULT_EDGE_ATTRIBUTES.keys())
        if type(attributes) != list:
            attributes = [attributes]
        for i, j, m in self.G.edges:
            for attribute in attributes:
                if attribute == 'weight':
                    self.G[i][j][m][attribute] = copy.deepcopy(
                        self.init_weights[self.G[i][j][m]['id']])
                else:
                    self.G[i][j][m][attribute] = copy.deepcopy(DEFAULT_EDGE_ATTRIBUTES[attribute])

    def _get_link_traffic(self):
        for i, j, m, edge_data in self.G.edges(keys=True, data=True):
            self.link_traffic[self.G[i][j][m]['id']] = edge_data['current_bandwidth'] / edge_data['bandwidth']

    def compute_reward_measure(self, measure=None, last=False):
        if measure is None:
            if self.reward_magnitude == 'link_traffic':
                measure = self.link_traffic
        if self.base_reward == 'mean_times_std':
            return np.mean(measure) * np.std(measure)
        elif self.base_reward == 'mean':
            return np.mean(measure)
        elif self.base_reward == 'std':
            return np.std(measure)
        elif self.base_reward == 'diff_min_max':
            return np.max(measure) - np.min(measure)
        elif self.base_reward == 'min_max':
            return np.max(measure)
        elif self.base_reward == 'phi':
            phi_value = self.phi(self.G, eval = self.eval_mode)
            self.phi_history.append(phi_value)
            return phi_value
        elif self.base_reward == 'fixed':
            phi_value = self.phi(self.G, eval = self.eval_mode)
            self.phi_history.append(phi_value)
            if phi_value == 0:
                return -0.1 if not last else -1
            return 0.1 if not last else 1

    def _compute_reward(self, current_reward_measure=None, last=False):
        if current_reward_measure is None:
            current_reward_measure = self.compute_reward_measure(last=last)
        if self.reward_computation == 'value':
            reward = - current_reward_measure
        elif self.reward_computation == 'change':
            reward = (self.reward_measure - current_reward_measure)
        elif self.reward_computation == 'change_with_bonus':
            bonus = 0
            if current_reward_measure < self.reward_measure:
                bonus = (0.1 - current_reward_measure) / (0.1 + current_reward_measure)
                bonus = 0 if bonus < 0 else bonus
            reward = self.reward_measure - current_reward_measure + bonus
        elif self.reward_computation == 'change_with_log_bonus':
            if current_reward_measure < self.reward_measure:
                bonus = min(0.005 / (math.log(current_reward_measure + 0.1, 10) + 1.01), 0.15)
            elif current_reward_measure > self.reward_measure:
                bonus = -min(0.005 / (math.log(self.reward_measure + 0.1, 10) + 1.01), 0.15)
            else:
                bonus = 0
            reward = self.reward_measure - current_reward_measure + bonus
        self.reward_measure = current_reward_measure
        return reward

    def get_hash_weights(self):
        hash_weights = HashWeights()
        topo_nodes = self.G.nodes()
        for edge_start, edge_end, edge_index, edge_data in self.G.edges(keys=True, data=True):
            # for now, weights for all destinations are the same
            for end_node in topo_nodes:
                if end_node == edge_start:
                    continue
                hash_weights.put(edge_start, end_node, edge_end, edge_index, edge_data["weight"])
        self.hash_weights = hash_weights
