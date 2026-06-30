from typing import Optional
from networkx import product
import networkx as nx
import numpy as np
import pandas as pd
import random
import os
from datetime import datetime

from dte_stand.config import Config

from dte_stand.data_structures import HashWeights, InputData
from dte_stand.hash_function import RandomHashFunction2
from dte_stand.phi_calculator import PhiCalculator
from dte_stand.paths import DAGCalculator
from dte_stand.logger import init_logger

import logging
LOG = logging.getLogger(__name__)

import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Phi Calculation on given weights matrix -- Parameters')

    parser.add_argument("dir", type=str,
        help="path to directory with topology.gml, flows.json")

    parser.add_argument("--mode", type=int, choices=[1, 2], default=1,
        help="1 -- find optimal weights for links, 2 -- find optimal weights for links x destinations (default: 1)")

    parser.add_argument("--start", type=int, default=0,
        help="time_start * flowsperiod will be the starting timestamp (default: 0)")

    parser.add_argument("--end", type=int,
        help="time_end * flowsperiod will be the ending timestamp (default: time_start + 1)")

    parser.add_argument('--flowsperiod', type=int, default=30000,
        help='Period of changes in input flows, ms. Should be equal to lsdb_period in MAROH\'s config, and value of period parameter in flow_generator.generate call in generator.py) (default: 30000)')

    parser.add_argument('--init_sol', type=str, default=None, help='initial solution file (npz)')

    parser.add_argument('--flowsname', type=str, default='flows.json',
        help='Name of file with flows to process (default: flows.json)')

    args = parser.parse_args()
    return args

class PhiTest:
    def __init__(self, problem_mode, topo, flows, path_calculator, phi_func,
                 experiment_path_rel, tim):
        self.problem_mode = problem_mode
        if self.problem_mode == 1:
            self.convert_to_hash_weights = self.convert_to_hash_weights_no_nexthops
        elif self.problem_mode == 2:
            self.convert_to_hash_weights = self.convert_to_hash_weights_with_nexthops
        else:
            raise ValueError("invalid problem mode")

        self.topo = topo
        self.n_nodes = len(list(self.topo.nodes))
        self.node_idx = dict(zip(
            list(self.topo.nodes), range(self.n_nodes)
        ))
        self.node_labels = dict(zip(
            range(self.n_nodes), list(self.topo.nodes)
        ))
        self.flows = flows
        self.path_calculator = path_calculator
        self.phi_func = phi_func
        self.experiment_path_rel = experiment_path_rel
        self.tim = tim

    def convert_to_hash_weights_no_nexthops(self, weight_matr):
        hash_weights = HashWeights()
        n = self.n_nodes
        for i in range(n):
            for j in range(n):
                if weight_matr[i][j] != 0:
                    for k in range(n):
                        hash_weights.put(self.node_labels[i], self.node_labels[k],
                            self.node_labels[j], 0, weight_matr[i][j])
        return hash_weights

    def convert_to_hash_weights_with_nexthops(self, weight_matr):
        hash_weights = HashWeights()
        n = self.n_nodes
        for i in range(n):
            for j in range(n):
                if weight_matr[i][j * n] != 0:
                    for k in range(n):
                        assert(weight_matr[i][j * n + k] != 0)
                        hash_weights.put(self.node_labels[i], self.node_labels[k],
                            self.node_labels[j], 0, weight_matr[i][j * n + k])
        return hash_weights

    def calc_phi(self, weights):
        hash_weights = self.convert_to_hash_weights(weights)
        topo_changed = self.topo.copy()
        hash_function = RandomHashFunction2(self.path_calculator)  # reset
        hash_function.run(topo_changed, self.flows, hash_weights)
        CBandwidth = pd.DataFrame(0, index=list(topo_changed.nodes),
                                  columns=list(topo_changed.nodes))
        for edge in topo_changed.edges(data=True):
            CBandwidth.loc[edge[0], edge[1]] += int(edge[-1]['current_bandwidth'])
        phi = self.phi_func(topo_changed)
        return phi, CBandwidth

    def run(self, population_file = None):
        if population_file is not None:
            weights = np.load(population_file)
        else:
            if self.problem_mode == 1:
                weights = np.zeros((self.n_nodes, self.n_nodes), dtype=int)
                for edge in self.topo.edges(data=True):
                    if edge[-1]['bandwidth'] != 0:
                        i = self.node_idx[edge[0]]
                        j = self.node_idx[edge[1]]
                        weights[i, j] = 1
            else:
                weights = np.zeros((self.n_nodes, self.n_nodes*self.n_nodes), dtype=int)
                for edge in self.topo.edges(data=True):
                    if edge[-1]['bandwidth'] != 0:
                        i = self.node_idx[edge[0]]
                        j = self.node_idx[edge[1]]
                        for k in range(self.n_nodes):
                            weights[i, j * self.n_nodes + k] = 1

        Bandwidth = pd.DataFrame(0, index=list(self.topo.nodes),
                                  columns=list(self.topo.nodes))

        for edge in self.topo.edges(data=True):
            assert(Bandwidth.loc[edge[0], edge[1]] == 0) # otherwise it's a multigraph, not supported yet
            Bandwidth.loc[edge[0], edge[1]] += edge[-1]['bandwidth']

        phi, cbw = self.calc_phi(weights)
        util = pd.DataFrame(0., index=cbw.index, columns=cbw.columns)
        # util_str = pd.DataFrame('', index=cbw.index, columns=cbw.columns)
        util_flatten = []
        for i in cbw.index:
            for j in cbw.columns:
                ut = (cbw.loc[i, j] / Bandwidth.loc[i, j]) if Bandwidth.loc[i, j] > 0 else np.nan
                util.loc[i, j] = ut
                # util_str.loc[i, j] = f"{ut:6.3f}"
                if Bandwidth.loc[i, j] > 0:
                    util_flatten.append(ut)
        util_max = np.max(util_flatten)
        util_mean = np.mean(util_flatten)

        return weights, phi, cbw, Bandwidth, util, util_mean, util_max

if __name__ == "__main__":
    # Output data will be written in root directory (using current timestamp to prevent overwriting).

    args = parse_args()

    # Reading topology
    experiment_folder = os.path.normpath(args.dir)
    experiment_path_rel = os.path.split(experiment_folder)[-1]
    problem_mode = args.mode
    tim_start = args.start
    tim_end = args.end if args.end is not None else tim_start + 1
    flowsperiod = args.flowsperiod

    dat_str = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

    Config.load_config(os.path.join(experiment_folder, '../..'))
    config = Config.config()
    flowsname_noext = os.path.splitext(args.flowsname)[0]
    log_filename = os.path.join(experiment_folder, f"{flowsname_noext}-{problem_mode}_{dat_str}.log")
    input_data = InputData(experiment_folder, flows_name=args.flowsname, ignore_topology_changes=True)
    topo = input_data.topology.get(0)[0]

    phi_func = PhiCalculator.calculate_phi
    path_calculator = DAGCalculator()

    path_calculator.prepare_iteration(topo)

    for tim in range(tim_start * flowsperiod, tim_end * flowsperiod, flowsperiod):
        print(f"env time: {tim}")
        flows = input_data.flows.get(tim)
        print(f"number of flows: {len(flows)}")
        filename_prefix = os.path.join(experiment_folder, f"{flowsname_noext}-ecmp-{problem_mode}-{tim}_{dat_str}")

        alg = PhiTest(problem_mode, topo, flows, path_calculator, phi_func, experiment_path_rel, tim)
        weights, phi, cbw, Bandwidth, util, util_mean, util_max = alg.run(population_file=args.init_sol)


        np.save(f'{filename_prefix}_population.npy', np.array([weights]))
        np.savez(f'{filename_prefix}_solution.npz', solution=weights, phi=np.array([phi]))

        # print(f"Weights matrix:\n {weights}")
        print(f"Phi function value: {phi:7.5f}")
        # print(f"Links load:\n {util}")
        print(f"Avg link load: {util_mean}")
        print(f"Max link load: {util_max}")
