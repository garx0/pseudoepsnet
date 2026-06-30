import random
import networkx
import copy
import itertools
from collections import defaultdict
from typing import Generator


def weight_func(u, v, attr):
    alpha = 0.0001  # very large weight if remained bandwidth is zero
    edge_dict = list(attr.values())[0]
    return 10000.0 / float(edge_dict['bandwidth'] + alpha)


class DemandMatrixGenerator:
    """
    Generator that creates random demand matrices for a given topology.
    These demand matrices can be used in flow generators
    """

    def __init__(self, min_bandwidth_coef: float, max_bandwidth_coef: float, topology: networkx.MultiDiGraph,
                     mode: str = 'gravity', seed: int = None):
        """
        :param min_bandwidth_coef: each demand will take at least this much fraction
                of remaining bandwidth between two nodes. Use this to control average network load
        :param max_bandwidth_coef: each demand will take at most this much fraction
                of remaining bandwidth between the nodes. Use this to control average network load
        :param topology: networkx topology
        :param mode: generating mode: 'gravity' (default), 'standard'
        :param seed: random seed
        """
        self._min_bandwidth = min_bandwidth_coef
        self._max_bandwidth = max_bandwidth_coef
        self._topology = topology
        self._mode = mode
        if self._mode == 'standard':
            self.generate = self.generate_standard
        elif self._mode == 'gravity':
            self.generate = self.generate_gravity
        else:
            raise ValueError("invalid mode name")
        self._seed = seed
        if self._seed is None:
            self._seed = random.randint(0, 65536)
        self._rng = random.Random(self._seed)
        self._shuffle_seed = self._rng.randint(0, 65536)
        random.seed(self._seed)

    def _get_topology_edges_between_nodes(
                self, topology: networkx.MultiDiGraph,
                node1: str, node2: str) -> Generator[tuple[int, dict], None, None]:
        n_edges = topology.number_of_edges(node1, node2)
        index = 0
        n_edges_found = 0
        while n_edges_found < n_edges:
            try:
                yield index, topology.edges[node1, node2, index]
            except KeyError:
                pass
            else:
                n_edges_found += 1
            index += 1

    def _convert_to_edge_path(
                self, topology: networkx.MultiDiGraph, path: list[str]) -> tuple[int, list[tuple[str, str, int]]]:
        # pairwise is not in itertools until python 3.10
        def pairwise(iterable):
            # pairwise('ABCDEFG') --> AB BC CD DE EF FG
            a, b = itertools.tee(iterable)
            next(b, None)
            return zip(a, b)

        edge_path: list[tuple[str, str, int]] = []
        min_bandwidth = -1
        for node1, node2 in pairwise(path):
            edges_between_nodes = self._get_topology_edges_between_nodes(topology, node1, node2)
            edge_index, edge_data = sorted(edges_between_nodes, key=lambda x: x[1]['bandwidth'], reverse=True)[0]
            if edge_data['bandwidth'] < min_bandwidth or min_bandwidth < 0:
                min_bandwidth = edge_data['bandwidth']
            edge_path.append((node1, node2, edge_index))
        return min_bandwidth, edge_path

    def _decrease_bandwidth(self, topology: networkx.MultiDiGraph, path: list[tuple[str, str, int]], demand: int):
        for node1, node2, index in path:
            topology.edges[node1, node2, index]['bandwidth'] -= demand

    def _calculate_average_bandwidth(self, topology: networkx.MultiDiGraph):
        number_of_edges = topology.number_of_edges()
        total_load = 0.0
        max_load = 0.0
        for node1, node2, index, edge_data in topology.edges(data=True, keys=True):
            # initial (total) bandwidth is stored in self._topology, current is stored in given topology
            cur_load = 1 - (edge_data['bandwidth'] / self._topology.edges[node1, node2, index]['bandwidth'])
            if cur_load > max_load:
                max_load = cur_load
            total_load += cur_load
        return total_load / number_of_edges, max_load

    def generate_standard(self, amount: int) -> tuple[list[dict[str, dict[str, int]]], float ,float]:
        """
        Main function to generate demand matrices
        :param amount: number of demand matrices to generate
        :return: list of demand matrices
        """
        matrices: list[dict[str, dict[str, int]]] = []
        for i in range(amount):
            topology = copy.deepcopy(self._topology)
            graph_nodes = list(topology.nodes())
            nodes_amount = len(graph_nodes)
            # random order of nodes (increasing their links' occupied bandwidth)
            shuffle_node_pairs = self._rng.sample(
                list(networkx.product.product(graph_nodes, graph_nodes)),
                nodes_amount * nodes_amount)
            demand_matrix: dict[str, dict[str, int]] = defaultdict(lambda: {})

            for node1, node2 in shuffle_node_pairs:
                if node1 == node2:
                    continue

                # find the shortest path through edges that have the most absolute remaining bandwidth
                shortest_path = networkx.shortest_path(topology, node1, node2, weight=weight_func)

                # convert to edge path and get the minimum link bandwidth on the path
                min_bandwidth, edge_path = self._convert_to_edge_path(topology, shortest_path)

                # take random fraction of the minimum bandwidth on the path
                demand = int(
                    self._rng.uniform(min_bandwidth * self._min_bandwidth,
                                   min_bandwidth * self._max_bandwidth)
                )
                print(
                    f"node1 = {node1}, node2 = {node2}, min_bandwidth = {min_bandwidth}, edge_path = {edge_path}, demand = {demand}")

                # decrease taken bandwidth from the path
                self._decrease_bandwidth(topology, edge_path, demand)

                # save demand
                demand_matrix[node1][node2] = demand
            avg, max_load = self._calculate_average_bandwidth(topology)
            print(f'(s) Iteration: {i}, average bandwidth taken: {avg}, max bandwidth taken: {max_load}')
            matrices.append(demand_matrix)
        return matrices, avg, max_load

    def _generate_unnorm_gravity_matrices(self, topology, matrix, cap_list, start_nodes, end_nodes):
        for node1 in start_nodes:
            for node2 in end_nodes:
                if node1 == node2:
                    continue

                # find the path with the most sum bandwidth
                # shortest_path = networkx.shortest_path(topology0, node1, node2, weight=weight_func)

                # find the shortest path (with least number of edges)
                shortest_path = networkx.shortest_path(topology, node1, node2, weight=lambda u, v, attr: 1)

                dist = len(shortest_path)

                node1_cap_out = cap_list[node1]  # src outgoing capacity
                node2_cap_in = cap_list[node2]  # dst ingoing capacity

                demand_unnorm: float = node1_cap_out * node2_cap_in / (dist * dist)  # demand (not normalized)
                # save demand
                matrix[node1][node2] = demand_unnorm

    def generate_gravity(self, amount: int,
                         start_nodes: list[str] = None, end_nodes: list[str] = None, n_interpolate: int = 1) -> tuple[list[dict[str, dict[str, int]]], float, float]:
        """
        Main function to generate demand matrices
        :param amount: number of demand matrices to generate
        :param start_nodes: list of nodes that generate traffic. Must not intersect with end_nodes.
        :param end_nodes: list of nodes that receive traffic. Must not intersect with start_nodes.
        :return: list of demand matrices
        """
        topology0 = copy.deepcopy(self._topology)
        graph_nodes = list(topology0.nodes())

        # outgoing capacity of a node (sum of capacities of outgoing links)
        # (ingoing capacity is always equal to outgoing)
        cap_list: dict[str, int] = {}

        for node in graph_nodes:
            out_bw = sum((
                edge_data['bandwidth'] for n1, n2, edge_data in topology0.out_edges([node], data=True)
            ))
            cap_list[node] = out_bw

        # base demand matrix, not normalized
        demand_matrix_unnorm_base: dict[str, dict[str, float]] = defaultdict(lambda: {})
        # base demand matrix
        demand_matrix_base: dict[str, dict[str, int]] = defaultdict(lambda: {})

        if start_nodes and end_nodes:
            self._generate_unnorm_gravity_matrices(
                    topology0, demand_matrix_unnorm_base, cap_list, start_nodes, end_nodes)
            self._generate_unnorm_gravity_matrices(
                    topology0, demand_matrix_unnorm_base, cap_list, end_nodes, start_nodes)
        elif not start_nodes and not end_nodes:
            self._generate_unnorm_gravity_matrices(
                    topology0, demand_matrix_unnorm_base, cap_list, graph_nodes, graph_nodes)
        else:
            print('both start_nodes and end_nodes must be set')
            exit(1)

        demand_unnorm_sum = sum((
            demand_matrix_unnorm_base[node1][node2]
            for node1 in demand_matrix_unnorm_base
            for node2 in demand_matrix_unnorm_base[node1] if node1 != node2
        ))

        # normalizing coefficient
        g: float = 2. * sum([cap_list[node] for node in graph_nodes]) / demand_unnorm_sum

        for node1 in demand_matrix_unnorm_base:
            for node2 in demand_matrix_unnorm_base[node1]:
                if node1 == node2:
                    # demand_matrix_base[node1][node2] = 0
                    continue
                demand = int(demand_matrix_unnorm_base[node1][node2] * g)
                demand_matrix_base[node1][node2] = demand

        matrices: list[dict[str, dict[str, int]]] = []

        # DEBUG
        import numpy as np
        node_idx = dict((node, i) for i, node in enumerate(topology0.nodes))
        demand_matrix_base_np = np.zeros((len(topology0.nodes), len(topology0.nodes)), dtype=np.int64)
        for node1 in demand_matrix_unnorm_base:
            for node2 in demand_matrix_unnorm_base[node1]:
                if node1 != node2:
                    demand_matrix_base_np[node_idx[node1], node_idx[node2]] = demand_matrix_base[node1][node2]

        # print(f"demand_matrix_base:\n{demand_matrix_base_np}\n")
        demand_matrix_before: dict[str, dict[str, int]] = None
        demand_matrix_after: dict[str, dict[str, int]] = defaultdict(lambda: {})
        for i in range(amount):
            topology = copy.deepcopy(self._topology)
            graph_nodes = list(topology.nodes())

            # random order of nodes
            # shuffle_rng = random.Random(self._shuffle_seed)
            # n_nodes = len(graph_nodes)
            # shuffle_node_pairs = shuffle_rng.sample(
            #     list(networkx.product.product(graph_nodes, graph_nodes)),
            #     n_nodes * n_nodes)
            node_pairs = list(networkx.product.product(graph_nodes, graph_nodes)) # not shuffled
            node_pairs_use = [(x, y) for x, y in node_pairs
                                      if x != y and x in demand_matrix_base
                                      and y in demand_matrix_base[x]]

            for node1, node2 in node_pairs_use:
                # calculate demand by randomly (element-wise) scaling the base demand matrix
                demand_base = demand_matrix_base[node1][node2]
                demand = int(
                    self._rng.uniform(self._min_bandwidth * demand_base, self._max_bandwidth * demand_base)
                )
                demand_matrix_after[node1][node2] = demand
            if demand_matrix_before is None:
                matrices.append(copy.deepcopy(demand_matrix_after))
                print("test pair demand:", demand_matrix_after[node_pairs_use[0][0]][node_pairs_use[0][1]])
            else:
                for i in range(1, n_interpolate+1):
                    demand_matrix = defaultdict(lambda: {})
                    coef = i / n_interpolate
                    for node1, node2 in node_pairs_use:
                        demand_matrix[node1][node2] = round(
                            (1 - coef) * demand_matrix_before[node1][node2] + \
                            coef * demand_matrix_after[node1][node2])
                    matrices.append(demand_matrix)
                    print("test pair demand:", demand_matrix[node_pairs_use[0][0]][node_pairs_use[0][1]])
            demand_matrix_before = demand_matrix_after
            demand_matrix_after = defaultdict(lambda: {})
        return matrices, 0., 0.
