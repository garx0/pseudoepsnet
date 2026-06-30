import numpy as np
from sklearn.cluster import MiniBatchKMeans, KMeans, AgglomerativeClustering
from random import random
import time
import logging
import os
import re
from collections import defaultdict

from types import SimpleNamespace

ACT_VEC_UPDATE = 0
ACT_VEC_NEW_UPDATE = 1
ACT_VEC_USE = 2
ACT_VEC_IGNORE = 3
ACT_VEC_USE_AND_UPDATE = 4

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
            print(f"initial_state is not None")
            self.mem_vec_full = initial_state[:, :, :state_size]
            self.act_vec_full = initial_state[:, :, state_size:-1]
            self.mem_vec = np.copy(self.mem_vec_full)
            self.act_vec = np.copy(self.act_vec_full)
            print(self.mem_vec.shape)
            print(self.act_vec.shape)
        else:
            print(f"initial_state is None")

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

class ClusterMemory(StatesMemory):
    def __init__(self, iteration_id, max_size, threshold, n_agents, state_size, n_actions, P, var, initial_state=None):
        super().__init__(iteration_id, max_size, max_size // 2, threshold, n_agents, state_size, n_actions, P, var, initial_state)


    def init_cluster(self, **kwargs):
        self.cluster = [MiniBatchKMeans(init='random', max_iter=100, max_no_improvement=10, reassignment_ratio=0.0, **kwargs)] * self.n_agents


    def defragmentate(self, pred, n_agent):
        _, idx = np.unique(pred[::-1], return_index=True, axis=0)
        idx = len(pred) - 1 - idx

        '''SAVE SOME RANDOM VALUES'''
        tmp = np.arange(len(pred) - 1)[~np.isin(np.arange(len(pred) - 1), idx)]
        idx = np.append(idx, np.random.choice(tmp, size=(self.max_size - len(idx)), replace=False), axis=0)

        to = np.arange(self.max_size)[~np.isin(np.arange(self.max_size), idx)]
        tmp = np.arange(self.max_size, self.max_size + self.max_new_size)
        fr = tmp[np.isin(tmp, idx)]

        self.mem_vec[to, n_agent, :] = self.mem_vec_new[fr - self.max_size, n_agent, :]
        self.act_vec[to, n_agent, :] = self.act_vec_new[fr - self.max_size, n_agent, :]


    def update_memory(self, new_states, update=True):
        '''
        '''
        if self.experience_gamma == 0.0:
            return ACT_VEC_IGNORE, None

        if self.mem_vec.shape[0] > 0:
            proximities = np.linalg.norm(self.mem_vec - new_states, axis=2)
            argmins = np.argmin(proximities, axis=0)
            mins = proximities[argmins, np.arange(self.n_agents)]
            if (random() < self.experience_gamma) and (np.sum(mins < self.threshold) == self.n_agents): # found close item, Action of closest obj
                return ACT_VEC_USE, self.act_vec[argmins, np.arange(self.mem_vec.shape[1]), :]

        if self.mem_vec.shape[0] < self.max_size:
            if update:
                self.mem_vec = np.append(self.mem_vec, [new_states], axis=0) # self.mem_vec.write(self.cur_size, new_states)
                return ACT_VEC_UPDATE, None # HAS TO WRITE ACTION AFTER ACTIONS
            else:
                return ACT_VEC_IGNORE, None

        if self.mem_vec_new.shape[0] < self.max_new_size:
            if update:
                self.mem_vec_new = np.append(self.mem_vec_new, [new_states], axis=0)
                return ACT_VEC_NEW_UPDATE, None # HAS TO WRITE ACTION AFTER ACTIONS TO ADDITIONAL
            else:
                return ACT_VEC_IGNORE, None

        if self.clustered == False:
            self.init_cluster(n_clusters=self.max_size)
            self.clustered = True
            for agent in range(self.n_agents):
                nodes = np.append(self.mem_vec, self.mem_vec_new, axis=0)[:, agent, :]
                pred = self.cluster[agent].fit_predict(nodes)
                self.defragmentate(pred, agent)
                self.init_cluster(n_clusters=self.max_size)
                # self.cluster[agent].fit(self.mem_vec[:, agent, :]) # Maybe not needed
        else:
            # t_start = time.time()
            for agent in range(self.n_agents):
                nodes = np.append(self.mem_vec, self.mem_vec_new, axis=0)[:, agent, :]
                if False:
                    pred = self.cluster[agent].fit_predict(nodes)
                else:
                    self.cluster[agent].partial_fit(nodes) #.partial_fit(self.mem_vec_new[:, agent, :])
                    pred = self.cluster[agent].predict(nodes)
                self.defragmentate(pred, agent)
                self.init_cluster(n_clusters=self.max_size)
                # self.cluster[agent].fit(self.mem_vec[:, agent, :]) # Maybe not needed
            # t_end = time.time()
            # print(f"Memory update algorithm took {t_end - t_start} sec.")

        del self.mem_vec_new
        del self.act_vec_new
        self.mem_vec_new = np.asarray([new_states])
        self.act_vec_new = np.empty([0, self.n_agents, 16]) # self.n_agents
        return ACT_VEC_NEW_UPDATE, None # HAS TO WRITE ACTION AFTER ACTIONS TO ADDITIONAL


    def update_action(self, new_acts):
        self.act_vec = np.append(self.act_vec, [new_acts], axis=0)


    def update_action_new(self, new_acts):
        self.act_vec_new = np.append(self.act_vec_new, [new_acts], axis=0)

class BypassMemory(StatesMemory):
    def __init__(self, iteration_id, max_size, threshold, n_agents, state_size, n_actions, P, var, initial_state=None):
        super().__init__(iteration_id, max_size, max_size // 2, threshold, n_agents, state_size, n_actions, P, var, initial_state)

    def update_memory(self, new_states, update=True):
        if update:
            self.mem_vec = np.append(self.mem_vec, [new_states], axis=0)
            return ACT_VEC_UPDATE, None # HAS TO WRITE ACTION AFTER ACTIONS
        else:
            return ACT_VEC_IGNORE, None

    def update_action(self, new_acts):
        self.act_vec = np.append(self.act_vec, [new_acts], axis=0)

    def update_action_new(self, new_acts):
        pass
