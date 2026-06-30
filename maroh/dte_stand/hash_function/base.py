import abc
import networkx
from networkx.exception import NodeNotFound, NetworkXNoPath
from typing import Generator, Optional, Any
from dte_stand.data_structures import HashWeights, Flow
from dte_stand.paths.base import BasePathCalculator
from dte_stand.data_structures import GraphPathElement, Bucket
from dte_stand.config import Config
from collections import deque

import logging
LOG = logging.getLogger(__name__)


class PathNotFoundError(Exception):
    def __init__(self, current_node: str, flow: Flow):
        message = f'Path not found at node {current_node} for flow {flow} from {flow.start} to {flow.end}'
        super().__init__(message)


class BaseHashFunction(metaclass=abc.ABCMeta):
    def __init__(self, path_calculator: BasePathCalculator, default_weight=1, debug_check_cycles=0):
        self.path_calculator = path_calculator
        self._default_weight = default_weight
        self._check_cycles = debug_check_cycles
        self._episodes_hash_weights: list[HashWeights] = []
        self._all_hash_weights: list[tuple[HashWeights, list[HashWeights]]] = []
        config = Config.config()
        self._store_hashweights = config.store_hashweights
        self.route_dct: dict[str, list[tuple[str, str, int]]] = {}

        self.split_flows = config.split_flows

    @property
    def all_hash_weights(self):
        return self._all_hash_weights

    def end_iteration(self):
        self.path_calculator.end_iteration()
        try:
            iteration_hash_weights = self._episodes_hash_weights.pop()
        except IndexError:
            # first iteration, no hash weights yet
            return
        self._all_hash_weights.append((iteration_hash_weights, self._episodes_hash_weights))
        self._episodes_hash_weights = []
        self.route_dct.clear()

    @abc.abstractmethod
    def _choose_nexthop(self, buckets: list[Bucket], flow_id: str,
                        use_flow_memory: bool = True, hash: int = None) -> Optional[GraphPathElement]:
        ...

    def _construct_bucket_list(self,
                               topology: networkx.MultiDiGraph,
                               flow_bandwidth: int,
                               nexthops: list[GraphPathElement],
                               all_bucket_edge_dict: dict[tuple[str, str, int], Bucket],
                               allow_overflow: bool = False) -> list[Bucket]:
        available_buckets: list[Bucket] = []
        added_nexthops = set()
        for nexthop in nexthops:
            nexthop_tuple = (nexthop.from_, nexthop.to_, nexthop.index)
            if nexthop_tuple in added_nexthops:
                continue
            added_nexthops.add(nexthop_tuple)

            nexthop_link = topology.edges[nexthop.from_, nexthop.to_, nexthop.index]
            if not allow_overflow and (nexthop_link['bandwidth'] - nexthop_link['current_bandwidth'] < flow_bandwidth):
                # putting this flow into this nexthop will overflow the link which is not allowed
                continue

            if nexthop_tuple in all_bucket_edge_dict:
                available_buckets.append(all_bucket_edge_dict[nexthop_tuple])
            else:
                available_buckets.append(Bucket(edge=nexthop, weight=self._default_weight))

        return available_buckets

    def _flow_path(self, topology: networkx.MultiDiGraph,
                   flow: Flow, hash_weights: HashWeights, current_node: str, previous_node: Optional[str],
                   depth: Optional[int] = None, use_flow_memory: bool = True, curr_len: int = 0) -> list[GraphPathElement]:
        if (current_node and current_node == flow.end) or (depth is not None and depth <= 0):
            return []

        try:
            nexthops = self.path_calculator.calculate(topology, current_node, previous_node, flow.end, flow.start, curr_len=curr_len)
        except NetworkXNoPath as e:
            raise PathNotFoundError(current_node, flow) from e
        except NodeNotFound:
            LOG.info(f'One of the nodes ({current_node} or {flow.end}) was removed from topology.'
                     f'Flow ({flow}) is dropped')
            raise

        # there might be a case when no bucket exists for a nexthop
        # it happens because hash weights come from previous iteration
        # and on previous iteration topology might have been different so the needed edge is not there
        # for such cases we create a bucket manually and set its weight to default weight
        # it probably (?) makes sense to set this weight as high as possible
        # because the edge is "new" and has no traffic
        all_bucket_edge_dict = {(bucket.edge.from_, bucket.edge.to_, bucket.edge.index): bucket
                                for bucket in hash_weights.get_bucket_list(current_node, flow.end)}
        available_buckets = self._construct_bucket_list(
                topology, flow.bandwidth, nexthops, all_bucket_edge_dict, allow_overflow=True)

        chosen_nexthop = self._choose_nexthop(available_buckets, flow.flow_id, use_flow_memory=use_flow_memory, hash=flow.hash)
        if not chosen_nexthop:
            raise PathNotFoundError(current_node, flow)

        flow_path = [chosen_nexthop]
        depth = depth - 1 if depth is not None else None
        flow_path.extend(
                self._flow_path(topology, flow, hash_weights, chosen_nexthop.to_, current_node,
                                depth=depth, use_flow_memory=use_flow_memory, curr_len=curr_len+1)
        )
        return flow_path

    def get_all_paths(self, topo: networkx.MultiDiGraph, start, end, bw, hash_weights: HashWeights):
        qr = deque([bw])
        q1 = deque([[start]])
        q2 = deque([[]])
        while q1:
            curr_node_path = q1.popleft()
            curr_edge_path = q2.popleft()
            curr_bw = qr.popleft()
            if len(curr_edge_path) > 0:
                last_edge = curr_edge_path[-1]
                topo.edges[last_edge.from_, last_edge.to_, last_edge.index]['current_bandwidth'] += curr_bw
            curr = curr_node_path[-1]
            if curr == end:
                continue
            prev = curr_node_path[-2] if len(curr_node_path) > 1 else None
            try:
                # print(f"{cnt}: {curr=}, {prev=}, {end=}, {start=}")
                try:
                    nexthops = self.path_calculator.calculate(topo, curr, prev, end, start)
                except:
                    raise
                all_bucket_edge_dict = {(bucket.edge.from_, bucket.edge.to_, bucket.edge.index): bucket
                                        for bucket in hash_weights.get_bucket_list(curr, end)}
                available_buckets = self._construct_bucket_list(
                        topo, bw, nexthops, all_bucket_edge_dict, allow_overflow=True)
                
            except:
                ways = []
            nexthop_tuples = []
            weight_sum = 0.0
            for bucket in available_buckets:
                weight_sum += bucket.weight
            for bucket in available_buckets:
                qr.append(curr_bw * bucket.weight / weight_sum)
                q1.append(curr_node_path + [bucket.edge.to_])
                q2.append(curr_edge_path + [bucket.edge])
        return

    def _check_cycle(self, path: list[GraphPathElement]):
        only_nodes = [elem.from_ for elem in path]
        only_nodes.append(path[-1].to_)
        only_nodes_set = set(only_nodes)

        if len(only_nodes_set) != len(only_nodes):
            LOG.warning(f'Cycle detected in path: {path}')

    def run(self, topology: networkx.MultiDiGraph, flows: list[Flow],
            hash_weights: HashWeights, link: Optional[tuple[str, str, int]] = None, depth: Optional[int] = None,
            use_flow_memory: bool = True) -> None:
        """
        Main function to run hash

        :param topology: current topology
        :param flows: current list of flows
        :param hash_weights: current hash weights
        :param link: link where the change occurred, tuple of (start node, end node, link index).
                     If None, reset the route dict and current bandwidth in topology
        :param depth: if None (default), full path will be found.
            If positive int, path finding will stop after <depth> hops. The rest of the path will not be calculated
                and bandwidths will not change
            If negative int or 0, all paths will be empty
        :param use_flow_memory: if True (default), hash function should use its flow_id map to remember
            already calculated nexthop for this flow (if the implementation has this ability)
        :return: None
        """
        if self._store_hashweights:
            self._episodes_hash_weights.append(hash_weights)

        new_route_flows = []
        if link is None:
            new_route_flows = flows
            self.route_dct.clear()
            for _, _, edge_data in topology.edges(data=True):
                edge_data['current_bandwidth'] = 0
        else:
            for flow in flows:
                try:
                    for t in self.route_dct[flow.flow_id]:
                        if link[0] == t[0]:
                            new_route_flows.append(flow)
                except KeyError:
                    new_route_flows.append(flow)

        if self.split_flows == True:
            for flow in new_route_flows:
                self.get_all_paths(topology, flow.start, flow.end, flow.bandwidth, hash_weights)
            self.path_calculator.end_flow_set()
            return

        for flow in new_route_flows:
            try:
                if link:
                    try:
                        for edge in self.route_dct[flow.flow_id]:
                            start_point, end_point, ind = edge
                            topology[start_point][end_point][ind]['current_bandwidth'] -= flow.bandwidth
                    except KeyError:
                        pass
                self.route_dct[flow.flow_id] = []

                flow_path = self._flow_path(topology, flow, hash_weights, flow.start, None,
                                            depth=depth, use_flow_memory=use_flow_memory)
            except PathNotFoundError:
                # LOG.exception('Failed to find path for flow: ')
                continue
            except NodeNotFound:
                # debug message is caught inside _flow_path
                continue
            if self._check_cycles:
                self._check_cycle(flow_path)
            for element in flow_path:
                topology.edges[element.from_, element.to_, element.index]['current_bandwidth'] += flow.bandwidth
                self.route_dct[flow.flow_id].append((element.from_, element.to_, element.index))

        self.path_calculator.end_flow_set()
