import networkx
from dte_stand.algorithm.base import BaseAlgorithm
from dte_stand.data_structures import HashWeights, Flow

import logging
LOG = logging.getLogger(__name__)


class DummyAlgorithm(BaseAlgorithm):
    def step(self, topology: networkx.MultiDiGraph, flows: list[Flow],
             iteration_num: int = 0, save_model: bool = False) -> HashWeights:
        LOG.debug('Running dummy algorithm')
        hash_weights = HashWeights()
        topo_nodes = topology.nodes
        for start_node in topo_nodes:
            for end_node in topo_nodes:
                if start_node == end_node:
                    continue
                try:
                    node_edges = list(topology.edges(nbunch=start_node, keys=True))
                except KeyError:
                    # node was removed from topology
                    continue
                for edge in node_edges:
                    edge_start, edge_end, edge_index = edge
                    hash_weights.put(edge_start, end_node, edge_end, edge_index, 1)
        return hash_weights


class UcmpDummyAlgorithm(BaseAlgorithm):
    def step(self, topology: networkx.MultiDiGraph, flows: list[Flow],
             iteration_num: int = 0, save_model: bool = False) -> HashWeights:
        hw = HashWeights()
        nodes = list(topology.nodes())
        for start, end, edge_data in topology.edges(data=True):
            weight = 1 - float(edge_data['current_bandwidth']) / edge_data['bandwidth']
            if weight < 0:
                weight = 0
            for dst_node in nodes:
                hw.put(start, dst_node, end, 0, weight)
        self._hash_function.run(topology, flows, hw, use_flow_memory=False)
        self._phi(topology)
        return hw
