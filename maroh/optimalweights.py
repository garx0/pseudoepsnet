from typing import Optional
from networkx import product
from sys import maxsize
import networkx as nx
import numpy as np
import pandas as pd
import random
import os
import sys
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
    parser = argparse.ArgumentParser(description='Genetic Algorithm Parameters')

    parser.add_argument("dir", type=str,
        help="path to directory with topology.gml, flows.json")

    parser.add_argument("--mode", type=int, choices=[1, 2], default=1,
        help="1 -- find optimal weights for links, 2 -- find optimal weights for links x destinations (default: 1)")

    parser.add_argument("--start", type=int, default=0,
        help="time_start * flowsperiod will be the starting timestamp (default: 0)")

    parser.add_argument("--end", type=int,
        help="time_end * flowsperiod will be the ending timestamp (default: time_start + 1)")

    parser.add_argument('--ntrain', type=int, default=1,
        help='Number of hash seeds for training (default: 1)')
    parser.add_argument('--ntest', type=int, default=1,
        help='Number of hash seeds for testing (default: 1)')
    parser.add_argument('--nvalid', type=int, default=50,
        help='Number of hash seeds for validation (default: 50)')

    parser.add_argument('--seed', type=int,
        help='Random seed for generating hash function seeds')
    parser.add_argument('--maxweight', type=int, default=64,
        help='Maximum weight (default: 64)')

    parser.add_argument('--flowsperiod', type=int, default=30000,
        help='Period of changes in input flows, ms. Should be equal to lsdb_period in MAROH\'s config, and value of period parameter in flow_generator.generate call in generator.py) (default: 30000)')

    parser.add_argument('--iter', type=int, default=100,
        help='Number of genetic iterations (default: 100)')
    parser.add_argument('--size', type=int, default=100,
        help='Size of the population (default: 100)')
    
    parser.add_argument('--init_sol', type=str, default=None, help='initial solution file (npz)')

    parser.add_argument('--crossrate', type=float, default=0.7,
        help='Crossover rate (default: 0.7)')
    parser.add_argument('--mutrate', type=float, default=0.7,
        help='Mutation rate (default: 0.7)')
    parser.add_argument('--swaps', type=int, default=5,
        help='Number of crossover swaps (default: 5)')
    parser.add_argument('--shifts', type=int, default=4,
        help='Number of mutation shifts (default: 4)')
    parser.add_argument('--reassigns', type=int, default=4,
        help='Number of mutation reassigns (how many weights to reassign in chosen row/column per mutation) (default: 4)')
    
    parser.add_argument('--flowsname', type=str, default='flows.json',
        help='Name of file with flows to process (default: flows.json)')

    args = parser.parse_args()
    return args

class GeneticAlgorithm:
    def __init__(self, problem_mode, topo, GENETIC_NUM, POPULATION_SIZE,
                 CROSSOVER_RATE, MUTATION_RATE, MAX_WEIGHT, CROSSOVER_SWAPS,
                 MUTATION_SHIFTS, MUTATION_REASSIGNS, flows, hash_seeds_train,
                 hash_seeds_test, hash_seeds_valid, path_calculator, phi_func,
                 filename, experiment_path_rel, tim):
        self.problem_mode = problem_mode
        if self.problem_mode == 1:
            self.crossover = self.crossover_no_nexthops
            self.mutation = self.mutation_no_nexthops
            self.convert_to_hash_weights = self.convert_to_hash_weights_no_nexthops
        elif self.problem_mode == 2:
            self.crossover = self.crossover_with_nexthops
            self.mutation = self.mutation_with_nexthops
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
        self.GENETIC_NUM = GENETIC_NUM
        self.POPULATION_SIZE = POPULATION_SIZE
        self.CROSSOVER_RATE = CROSSOVER_RATE
        self.MUTATION_RATE = MUTATION_RATE
        self.MAX_WEIGHT = MAX_WEIGHT
        self.CROSSOVER_SWAPS = CROSSOVER_SWAPS
        self.MUTATION_SHIFTS = MUTATION_SHIFTS
        self.MUTATION_REASSIGNS = MUTATION_REASSIGNS
        self.flows = flows
        self.hash_seeds_train = hash_seeds_train
        self.hash_seeds_test = hash_seeds_test
        self.hash_seeds_valid = hash_seeds_valid
        self.path_calculator = path_calculator
        self.phi_func = phi_func
        self.filename = filename
        self.experiment_path_rel = experiment_path_rel
        self.tim = tim

        LOG.debug(f"GENETIC_NUM_ITERATIONS: {self.GENETIC_NUM}")
        LOG.debug(f"POPULATION_SIZE: {self.POPULATION_SIZE}")
        LOG.debug(f"MAX_WEIGHT: {self.MAX_WEIGHT}")
        LOG.debug(f"CROSSOVER_RATE: {self.CROSSOVER_RATE}")
        LOG.debug(f"MUTATION_RATE: {self.MUTATION_RATE}")
        LOG.debug(f"CROSSOVER_SWAPS: {self.CROSSOVER_SWAPS}")
        LOG.debug(f"MUTATION_SHIFTS: {self.MUTATION_SHIFTS}")
        LOG.debug(f"MUTATION_REASSIGNS: {self.MUTATION_REASSIGNS}")
        LOG.debug(f"NUM_HASH_SEEDS_TRAIN: {len(self.hash_seeds_train)}")
        LOG.debug(f"NUM_HASH_SEEDS_TEST: {len(self.hash_seeds_test)}")
        LOG.debug(f"NUM_HASH_SEEDS_VALID: {len(self.hash_seeds_valid)}")
        LOG.debug(f"PROBLEM_MODE: {self.problem_mode}")

    def population_generation(self, cost):
        """
        Random population generation
        """
        population = []
        for i in range(self.POPULATION_SIZE):
            population.append(np.zeros_like(cost, dtype=np.float32))
        for i in range(cost.shape[0]):
            for j in range(cost.shape[1]):
                if cost[i,j] != 0:
                    for k in range(self.POPULATION_SIZE):
                        population[k][i][j] = np.random.rand() * 0.99 + 0.01
        return population

    def selection_direct(self, population, phi_list):
        """
        Selection using roulette wheel with probability ~ 1 / fitness
        """
        choice_weights = np.array(list(map(lambda x: 1/x, phi_list)))
        choice_weights /= np.sum(choice_weights)
        new_population_idx = np.random.choice(np.arange(len(population), dtype=int), self.POPULATION_SIZE, p=choice_weights)
        new_population = [population[i] for i in new_population_idx]
        population.clear()
        population.extend(new_population)

    def selection(self, population, phi_list):
        """
        Selection using linear ranking
        """
        n = len(phi_list)
        ranks = np.argsort(np.argsort(phi_list))
        sp = 1.9 # sp in [1,2]. the bigger is sp, the more different are weights
        linear_ranking = lambda i: (sp - (2 * sp - 2) * i / (n-1)) / n
        p = list(map(linear_ranking, ranks))
        new_population_idx = np.random.choice(np.arange(len(population), dtype=int), self.POPULATION_SIZE, p=p)
        new_population = [population[i] for i in new_population_idx]
        population.clear()
        population.extend(new_population)

    def crossover_no_nexthops(self, population):
        """
        Crossover
        """
        num = len(population) - (len(population) % 2)

        for i in range(0, num, 2):
            if np.random.rand() <= self.CROSSOVER_RATE:
                child1: np.ndarray = population[i].copy()
                child2: np.ndarray = population[i+1].copy()
                comparison: np.ndarray = (child1 == child2)
                if comparison.all():
                    continue
                for it in range(self.CROSSOVER_SWAPS):
                    crossover_row = np.random.randint(0, len(child1))
                    if np.random.rand() <= 0.5:
                        # doing for row
                        t1 = child1[crossover_row].copy()
                        t2 = child2[crossover_row].copy()
                        child1[crossover_row] = t2
                        child2[crossover_row] = t1
                    else:
                        # doing for column
                        t3 = child1[:, crossover_row].copy()
                        t4 = child2[:, crossover_row].copy()
                        child1[:, crossover_row] = t4
                        child2[:, crossover_row] = t3
                population.append(child1)
                population.append(child2)

    def crossover_with_nexthops(self, population):
        """
        Crossover
        """
        self.crossover_no_nexthops(population)

    def mutation_no_nexthops(self, population):
        """
        Shift mutation
        """
        for i in range(0, len(population)):
            if np.random.rand() <= self.MUTATION_RATE:
                new_s = population[i].copy()
                n = np.random.randint(0, new_s.shape[0])
                for it in range(self.MUTATION_SHIFTS):
                    # cyclic shift the row's (or column's) non-zero elements by one element,
                    # and reassign its random elements to new random values;
                    rowwise = (np.random.rand() <= 0.5)
                    if not rowwise:
                        new_s = new_s.T
                    idx_nonzero = np.where(new_s[n] != 0)
                    if np.random.rand() <= 0.5:
                        # shift
                        new_s[n][idx_nonzero] = np.roll(new_s[n][idx_nonzero], -1)
                    for it2 in range(self.MUTATION_REASSIGNS):
                        m = np.random.randint(0, len(idx_nonzero))
                        new_s[n][idx_nonzero[m]] = np.random.rand() * 0.99 + 0.01
                    if not rowwise:
                        new_s = new_s.T
                population.append(new_s)

    def mutation_with_nexthops(self, population):
        """
        Shift mutation
        """
        for i in range(0, len(population)):
            if np.random.rand() <= self.MUTATION_RATE:
                new_s = population[i].copy()
                n = np.random.randint(0, new_s.shape[0])
                for it in range(self.MUTATION_SHIFTS):
                    # cyclic shift the row's (or column's) non-zero elements by one element,
                    # and reassign its random elements to new random values;
                    rowwise = (np.random.rand() <= 0.5)
                    if not rowwise:
                        new_s = new_s.T
                    idx_nonzero = np.where(new_s[n] != 0)
                    if np.random.rand() <= 0.5:
                        # shift
                        new_s[n][idx_nonzero] = np.roll(new_s[n][idx_nonzero], -1)
                    for it2 in range(self.MUTATION_REASSIGNS * (n if rowwise else 1)):
                        m = np.random.randint(0, len(idx_nonzero))
                        new_s[n][idx_nonzero[m]] = np.random.rand() * 0.99 + 0.01
                    if not rowwise:
                        new_s = new_s.T
                population.append(new_s)

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

    def calc_phi(self, hash_weights, hash_seed=123):
        topo_changed = self.topo.copy()
        hash_function = RandomHashFunction2(self.path_calculator)  # reset
        hash_function.run(topo_changed, self.flows, hash_weights)
        CBandwidth = pd.DataFrame(0, index=list(topo_changed.nodes),
                                  columns=list(topo_changed.nodes))
        for edge in topo_changed.edges(data=True):
            CBandwidth.loc[edge[0], edge[1]] += int(edge[-1]['current_bandwidth'])
        phi = self.phi_func(topo_changed)
        return phi, CBandwidth

    def calc_phi_multi(self, weights, hash_seeds):
        hash_weights = self.convert_to_hash_weights(weights)
        phi_tries = []
        cbw_tries = []
        CBandwidth = None
        for hash_seed in hash_seeds:
            phi, CBandwidth = self.calc_phi(hash_weights, hash_seed=hash_seed)
            cbw_tries.append(CBandwidth)
            phi_tries.append(phi)
        return phi_tries, cbw_tries

    def phi_stats(self, phi_list):
        phi_mean = np.mean(phi_list)
        phi_std = np.std(phi_list)
        return phi_mean, phi_std

    def cbw_stats(self, cbw_list):
        CBandwidth_example = cbw_list[0]
        cbw_list_np = np.array([CBandwidth.to_numpy() for CBandwidth in cbw_list])

        CBandwidth_mean = pd.DataFrame(np.mean(cbw_list_np, axis=0), index=CBandwidth_example.index,
                                       columns=CBandwidth_example.columns)
        CBandwidth_std = pd.DataFrame(np.std(cbw_list_np, axis=0), index=CBandwidth_example.index,
                                      columns=CBandwidth_example.columns)
        return CBandwidth_mean, CBandwidth_std

    def fitness(self, population, hash_seeds):
        phi_lists = []
        phi_mean_list = []
        phi_std_list = []
        cbw_lists = [] # current bandwidths dataframe for each population (mean over hash_seed tries)
        cbw_mean_list = []
        cbw_std_list = []
        for weights in population:
            phi_list, cbw_list = self.calc_phi_multi(weights, hash_seeds)
            phi_mean, phi_std = self.phi_stats(phi_list)
            CBandwidth_mean, CBandwidth_std = self.cbw_stats(cbw_list)
            phi_lists.append(phi_list)
            cbw_lists.append(cbw_list)
            phi_mean_list.append(phi_mean)
            phi_std_list.append(phi_std)
            cbw_mean_list.append(CBandwidth_mean)
            cbw_std_list.append(CBandwidth_std)
        return phi_lists, phi_mean_list, phi_std_list, cbw_lists, cbw_mean_list, cbw_std_list

    def run(self, population_file = None):
        if self.problem_mode == 1:
            cost = np.zeros((self.n_nodes, self.n_nodes), dtype=int)
            for edge in self.topo.edges(data=True):
                if edge[-1]['bandwidth'] != 0:
                    i = self.node_idx[edge[0]]
                    j = self.node_idx[edge[1]]
                    cost[i, j] = 1
        else:
            cost = np.zeros((self.n_nodes, self.n_nodes*self.n_nodes), dtype=int)
            for edge in self.topo.edges(data=True):
                if edge[-1]['bandwidth'] != 0:
                    i = self.node_idx[edge[0]]
                    j = self.node_idx[edge[1]]
                    for k in range(self.n_nodes):
                        cost[i, j * self.n_nodes + k] = 1

        population = self.population_generation(cost)

        if population_file is not None:
            population = np.load(population_file)
            population = list(population)

        Bandwidth = pd.DataFrame(0, index=list(self.topo.nodes),
                                  columns=list(self.topo.nodes))

        for edge in self.topo.edges(data=True):
            assert(Bandwidth.loc[edge[0], edge[1]] == 0) # otherwise it's a multigraph, not supported yet
            Bandwidth.loc[edge[0], edge[1]] += edge[-1]['bandwidth']

        optimal_value = 1e10
        optimal_sol = population[0]
        optimal_value_std = 0
        optimal_value_test_mean = 0
        optimal_value_test_std = 0
        optimal_sol_cbw_str = (Bandwidth/Bandwidth).astype(str)
        phi_mean_list = []
        n_phi_calcs = 0
        flowset_time_elapsed = 0.0
        min_phi = maxsize
        plot_data = []

        """
        GA algorithm steps
        """
        try:
            for it in range(self.GENETIC_NUM+1):
                t1 = datetime.now()
                LOG.info(f"GA iteration {it} / {self.GENETIC_NUM}...")
                if it > 0:
                    self.selection(population, phi_mean_list)
                    self.crossover(population)
                    self.mutation(population)
                phi_lists, phi_mean_list, phi_std_list, \
                    cbw_lists, cbw_mean_list, cbw_std_list = self.fitness(population, self.hash_seeds_train)
                n_phi_calcs += len(population) * len(self.hash_seeds_train)
                t2 = datetime.now()
                flowset_time_elapsed += (t2-t1).total_seconds()
                min_sol_idx = np.argmin(phi_mean_list)
                min_phi = phi_mean_list[min_sol_idx]
                min_phi_list = phi_lists[min_sol_idx]
                min_sol = population[min_sol_idx]
                min_phi_std = phi_std_list[min_sol_idx]
                if hash_seeds_test:
                    min_phi_test_list, _ = self.calc_phi_multi(min_sol, self.hash_seeds_test)
                else:
                    min_phi_test_list = [0]
                min_phi_test_mean, min_phi_test_std = self.phi_stats(min_phi_test_list)
                min_cbw_mean = cbw_mean_list[min_sol_idx]
                min_cbw_std = cbw_std_list[min_sol_idx]
                min_cbw_str = pd.DataFrame('', index=min_cbw_mean.index,
                                                columns=min_cbw_mean.columns)
                for i in min_cbw_str.index:
                    for j in min_cbw_mean.columns:
                        x1 = (min_cbw_mean.loc[i, j] / Bandwidth.loc[i, j]) if Bandwidth.loc[i, j] > 0 else np.nan
                        x2 = (min_cbw_std.loc[i, j] / Bandwidth.loc[i, j]) if Bandwidth.loc[i, j] > 0 else np.nan
                        min_cbw_str.loc[i, j] = f"{x1:6.3f}±{x2:5.3f}"
                if min_phi < optimal_value:
                    optimal_value = min_phi
                    optimal_sol = min_sol
                    optimal_value_std = min_phi_std
                    optimal_value_test_mean = min_phi_test_mean
                    optimal_value_test_std = min_phi_test_std
                    optimal_sol_cbw_str = min_cbw_str
                if it == 0:
                    np.save(f'{self.filename}_iter0_population.npy', np.array(phi_mean_list))
                LOG.debug(f"{it:2}:\n best weight matrix in population:\n{min_sol}")
                LOG.info(f"{it:2}: best phi = {min_phi:7.5f}±{min_phi_std:6.4f} (for other hash functions: {min_phi_test_mean:6.4f}±{min_phi_test_std:6.4f})")
                phi_min_train_list_str = ', '.join([f"{phi:.8f}" for phi in min_phi_list])
                phi_min_test_list_str = ', '.join([f"{phi:.8f}" for phi in min_phi_test_list])
                LOG.debug(f"{it:2}: best phi (all hash functions): [{phi_min_train_list_str}] (for other hash functions: [{phi_min_test_list_str}])")
                print(it, phi_min_train_list_str)
                LOG.debug(f"links load:\n{min_cbw_str}")
                LOG.debug(f"population len = {len(population)}")
                # phi_list_str = " ".join([f"{phi_mean:6.4f}±{phi_std:6.4f}, "
                #                          for phi_mean, phi_std in zip(phi_mean_list, phi_std_list)]).ljust(150 * 11)
                # LOG.debug(f"the whole population: {phi_list_str}") # uncomment these to print the whole population
                LOG.debug("")
                LOG.debug("")
                for phi in min_phi_list:
                    plot_data.append((self.experiment_path_rel, len(self.flows),
                        self.problem_mode, self.tim, it, phi, "train"))
                for phi in min_phi_test_list:
                    plot_data.append((self.experiment_path_rel, len(self.flows),
                        self.problem_mode, self.tim, it, phi, "test"))
        except KeyboardInterrupt:
            LOG.info("================ KeyboardInterrupt")

        LOG.info(f"Optimal solution:\n {optimal_sol}")
        LOG.info(f"Phi-function value: {optimal_value:7.5f}±{optimal_value_std:7.5f}"
                 f" (for other hash functions: {optimal_value_test_mean:7.5f}±{optimal_value_test_std:7.5f})")
        print(f"Phi-function value: {optimal_value:7.5f}±{optimal_value_std:7.5f}")
        LOG.info(f"links load:\n {optimal_sol_cbw_str}")
        min_phi_list, _ = self.calc_phi_multi(optimal_sol, self.hash_seeds_train)
        min_phi_test_list, _ = self.calc_phi_multi(optimal_sol, self.hash_seeds_valid)
        for phi in min_phi_list:
            plot_data.append((self.experiment_path_rel, len(self.flows),
                self.problem_mode, self.tim, -1, phi, "train"))
        for phi in min_phi_test_list:
            plot_data.append((self.experiment_path_rel, len(self.flows),
                self.problem_mode, self.tim, -1, phi, "valid"))
        columns = ["experiment", "n_flows", "problem_mode", "env_time", "iteration", "phi_min", "hash_funcs_set"]
        df = pd.DataFrame(plot_data, columns=columns)
        df.to_csv(f"{self.filename}.csv", index=False)
        # np.save(f'{self.filename}_solution.npy', optimal_sol)

        np.save(f'{self.filename}_population.npy', np.array(population))
        np.savez(f'{self.filename}_solution.npz', solution=optimal_sol, phi=np.array([optimal_value]))
        stats = (optimal_value_std, optimal_value_test_mean, optimal_value_test_std,
            optimal_sol_cbw_str, flowset_time_elapsed, n_phi_calcs)
        return optimal_sol, optimal_value, plot_data, stats

if __name__ == "__main__":
    # Genetic algorithm (GA) will use average of Ф value over the set of
    # training hash functions as a METRIC of a candidate.
    # It will use this metric to rank candidates in a population and select best candidate.
    # After each iteration Ф will be also calculated (but not used)
    # on the set of testing hash functions for the currently best candidate.
    # After the last iteration, Ф values will be calculated (but not used) on the set
    # of validation hash functions for the overall best candidate.
    # Thus, Ф is calculated:
    # - for the WHOLE population on EVERY iteration of GA on every TRAINING hash function;
    # - once per GA iteration on every TESTING hash function;
    # - once after the last GA iteration on every VALIDATION hash function.
    # Choose number of training, testing and validation functions considering how long
    # it will take to calculate Ф that many times.

    # Output data will be written in root directory (using current timestamp to prevent overwriting).

    args = parse_args()

    NUM_HASH_SEEDS_TRAIN = args.ntrain
    NUM_HASH_SEEDS_TEST = args.ntest
    NUM_HASH_SEEDS_VALID = args.nvalid
    RANDOM_SEED = args.seed if args.seed is not None else random.randint(0, 1000)

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
    # verbose info in log file, less info in stdout
    # init_logger(log_filename, "DEBUG", ['matplotlib'], "INFO")
    input_data = InputData(experiment_folder, flows_name=args.flowsname, ignore_topology_changes=True)
    topo = input_data.topology.get(0)[0]

    LOG.debug(f"RANDOM_SEED: {RANDOM_SEED}")
    LOG.debug(f"EXPERIMENT_FOLDER: {experiment_folder}")
    LOG.debug(f"TIME_START: {tim_start}")
    LOG.debug(f"TIME_END: {tim_end}")
    LOG.debug(f"FLOWS_PERIOD: {flowsperiod}")

    np.random.seed(RANDOM_SEED)
    random.seed(RANDOM_SEED)

    phi_func = PhiCalculator.calculate_max_util
    path_calculator = DAGCalculator()

    # first training seed (None) will be same as in usual MAROH run
    hash_seeds_traintest = [None] + [RANDOM_SEED*(i+1)
        for i in range(NUM_HASH_SEEDS_TRAIN + NUM_HASH_SEEDS_TEST + NUM_HASH_SEEDS_VALID - 1)]
    hash_seeds_train = hash_seeds_traintest[:NUM_HASH_SEEDS_TRAIN]
    hash_seeds_test = hash_seeds_traintest[NUM_HASH_SEEDS_TRAIN:NUM_HASH_SEEDS_TRAIN+NUM_HASH_SEEDS_TEST]
    hash_seeds_valid = hash_seeds_traintest[-NUM_HASH_SEEDS_VALID:]

    path_calculator.prepare_iteration(topo)

    for tim in range(tim_start * flowsperiod, tim_end * flowsperiod, flowsperiod):
        LOG.info(f"env time: {tim}")
        flows = input_data.flows.get(tim)
        LOG.debug(f"number of flows: {len(flows)}")
        filename_prefix = os.path.join(experiment_folder, f"{flowsname_noext}-{problem_mode}-{tim}_{dat_str}")
        t1_0 = datetime.now()

        gen_alg = GeneticAlgorithm(problem_mode, topo, args.iter, args.size,
                     args.crossrate, args.mutrate, args.maxweight, args.swaps,
                     args.shifts, args.reassigns, flows, hash_seeds_train,
                     hash_seeds_test, hash_seeds_valid, path_calculator, phi_func,
                     filename_prefix, experiment_path_rel, tim)
        optimal_sol, optimal_value, plot_data, stats = gen_alg.run(population_file=args.init_sol)
        optimal_value_std, optimal_value_test_mean, optimal_value_test_std, \
            optimal_sol_cbw_str, flowset_time_elapsed, n_phi_calcs = stats

        t2_0 = datetime.now()
        LOG.info(f"time elapsed (total): {(t2_0-t1_0).total_seconds():.6f} s", )
        LOG.info(f"time elapsed (clean): {flowset_time_elapsed:.6f} s") # only on fitness function
        LOG.info(f"({flowset_time_elapsed / (gen_alg.GENETIC_NUM+1):.6f} s / iter, on {gen_alg.GENETIC_NUM+1} iterations)")
        LOG.info(f"({flowset_time_elapsed / n_phi_calcs:.6f} s / weight matrix, on {n_phi_calcs} matrices)")
        avg_population_size = float(n_phi_calcs) / (gen_alg.GENETIC_NUM+1) / len(gen_alg.hash_seeds_train)
        LOG.info(f"population size (parameter) = {gen_alg.POPULATION_SIZE}, actual average size = {avg_population_size:.2f}")
