import networkx as nx
import itertools
from typing import Optional, Iterable
from collections import defaultdict
from contextlib import contextmanager
from networkx.algorithms.shortest_paths import shortest_path_length
from networkx.algorithms.traversal import dfs_tree, dfs_edges
from networkx.algorithms.dag import dag_longest_path
from typing import Generator

from dte_stand.paths.base import BasePathCalculator
from dte_stand.data_structures import GraphPathElement

import logging
LOG = logging.getLogger(__name__)


def dfs_edges_ignore(G, source=None, depth_limit=None, ignore: Iterable | None = None):
    """
    Networkx implementation of dfs_edges with ability to ignore some vertices
    @param ignore: list of vertices to ignore
    """
    ignore = ignore if ignore else []
    if source is None:
        # edges for all components
        nodes = G
    else:
        # edges for components with source
        nodes = [source]
    if depth_limit is None:
        depth_limit = len(G)

    visited = set()
    for start in nodes:
        if start in ignore or start in visited:
            continue
        visited.add(start)
        stack = [(start, iter(G[start]))]
        depth_now = 1
        while stack:
            parent, children = stack[-1]
            for child in children:
                if child in ignore:
                    continue
                if child not in visited:
                    yield parent, child
                    visited.add(child)
                    if depth_now < depth_limit:
                        stack.append((child, iter(G[child])))
                        depth_now += 1
                        break
            else:
                stack.pop()
                depth_now -= 1


class DAGCalculator(BasePathCalculator):
    def __init__(self, length_cutoff_fraction=2):
        super().__init__()
        self._forward_ordering = nx.MultiDiGraph()
        self._reverse_ordering = nx.MultiDiGraph()
        # how much longer the found paths can be compared to the shortest hop path
        self._length_cutoff = length_cutoff_fraction
        # keep a global track of nodes currently removed from topology
        self._forward_removed_nodes = set()
        self._reverse_removed_nodes = set()
        # precalculated path lengths to speed up the route construction
        self._forward_path_lengths: dict[str, dict[str, dict[tuple, int]]] = defaultdict(lambda: defaultdict(lambda: {}))
        self._reverse_path_lengths: dict[str, dict[str, dict[tuple, int]]] = defaultdict(lambda: defaultdict(lambda: {}))
        self._min_path_lengths: dict[str, dict[str, int]] = {}

        self.gml_dict = {}
        self.hash_paths_dict = {}

    def _get_topology_edges_between_nodes(
                self, topology: nx.MultiDiGraph,
                node1: str, node2: str) -> Generator[tuple[int, dict], None, None]:
        index = 0
        while True:
            try:
                yield index, topology.edges[node1, node2, index]
            except KeyError:
                # end of edges
                break
            index += 1

    def _node_with_longest_path(self, topology: nx.Graph) -> str:
        max_path_length = 0
        max_len_node = None
        for node in topology.nodes():
            dag_topology = dfs_tree(topology, node)
            longest_path = dag_longest_path(dag_topology)
            new_length = len(longest_path)
            if new_length > max_path_length:
                max_path_length = new_length
                max_len_node = node
        return max_len_node

    def _save_path_length(self, graph: nx.MultiDiGraph, source: str, destination: str,
                          removed_tuple: tuple) -> None:
        with self._hide_nodes(graph, removed_tuple):
            try:
                length = shortest_path_length(graph, source, destination)
            except nx.NetworkXNoPath:
                return
            self.get_lengths(graph)[source][destination][removed_tuple] = length

    # NOTE: current version is fast enough for small graph with 16 nodes
    # For larger graphs it will be slow.
    # Further improvements are possible:
    # - use Dijkstra one-to-many;
    # - use Johnson algorithm for finding all shortest paths;
    # - replace removal of nodes with some labelling logic;
    # - consider possibility to decrease number of removed nodes;
    # - store results in filesystem;
    # - rewrite in another language;

    def _calculate_min_path_lengths(self, graph: nx.MultiDiGraph) -> None:
        all_nodes = sorted(list(graph.nodes()))
        for source, destination in itertools.product(all_nodes, repeat=2):
            try:
                length = shortest_path_length(graph, source, destination)
                if source not in self._min_path_lengths:
                    self._min_path_lengths[source] = {}
                self._min_path_lengths[source][destination] = length
            except nx.NetworkXNoPath:
                continue

    def _calculate_path_lengths(self, graph: nx.MultiDiGraph, remove_max: int = 3) -> None:
        """
        Calculate and save point-to-point path lengths:
         - for all pairs of nodes
         - for all possible sets of removed nodes with given max size
        :param graph: either self._forward_ordering or self._reverse_ordering
        :param remove_max: max number of nodes to remove
        :return: None
        """
        lengths = self._forward_path_lengths if graph is self._forward_ordering else self._reverse_path_lengths

        def remove_rec(path, length, removed_tuple):
            """
            Recalculate path if last removed vertex is in path
            Update path lengths
            Run recursively with new nodes to remove
            """
            if (source in removed_tuple) or (destination in removed_tuple):
                return
            
            if len(removed_tuple) == 0 or removed_tuple[-1] in path:
                with self._hide_nodes(graph, removed_tuple):
                    try:
                        path = nx.shortest_path(graph, source, destination)
                        length = len(path) - 1
                    except nx.NetworkXNoPath:
                        return
            
            removed_tuple = tuple(sorted(removed_tuple))
            lengths[source][destination][removed_tuple] = length

            if len(removed_tuple) < remove_max:
                for v in reversed(all_nodes):
                    if len(removed_tuple) != 0 and v <= removed_tuple[-1]:
                        # use ordering to skip the same combinations
                        # relies on all nodes and removed tuple to be sorted
                        break
                    remove_rec(path, length, removed_tuple + (v,))

        all_nodes = sorted(list(graph.nodes()))
        for source, destination in itertools.product(all_nodes, repeat=2):
            path = []
            length = 0
            remove_rec(path, length, ())

    def _calculate_path_lengths_slow(self, graph: nx.MultiDiGraph) -> None:
        """
        Fill the self._forward_path_lengths and self._reverse_path_lengths with data

        Precalculate path lengths for each pair of nodes for each possible set of removed nodes.
        This is useful during path calculation when instead of doing shortest_path_length every time,
            we can look into this precalculated dict instead
        :param graph: either self._forward_ordering or self._reverse_ordering
        :return: None
        """
        all_nodes = list(graph.nodes())
        for source, destination in itertools.product(all_nodes, repeat=2):
            for removed1, removed2, removed3 in itertools.combinations(all_nodes, 3):
                removed_tuple = tuple(sorted((removed1, removed2, removed3)))
                if (source in removed_tuple) or (destination in removed_tuple):
                    continue
                self._save_path_length(graph, source, destination, removed_tuple)

            for removed1, removed2 in itertools.combinations(all_nodes, 2):
                removed_tuple = tuple(sorted((removed1, removed2)))
                if (source in removed_tuple) or (destination in removed_tuple):
                    continue
                self._save_path_length(graph, source, destination, removed_tuple)

            for removed_node in all_nodes:
                if (source == removed_node) or (destination == removed_node):
                    continue
                self._save_path_length(graph, source, destination, (removed_node,))

            self._save_path_length(graph, source, destination, tuple())

    def prepare_iteration(self, topology: nx.MultiDiGraph) -> None:
        # TODO: if topology did not change, do not recalculate graphs
        #   or somehow pass the topology changes here
        if self.gml in self.gml_dict:
            self._forward_ordering, self._reverse_ordering, self._forward_path_lengths, self._reverse_path_lengths, self._min_path_lengths, self.hash_paths = self.gml_dict[self.gml]
            return
        
        self.hash_paths = {}

        gml = self.gml
        paths = self.hash_paths
        gml_dict = self.gml_dict
        self.__init__()
        self.gml = gml
        self.gml_dict = gml_dict
        self.hash_paths = paths

        self._forward_path_lengths: dict[str, dict[str, dict[tuple, int]]] = defaultdict(lambda: defaultdict(lambda: {}))
        self._reverse_path_lengths: dict[str, dict[str, dict[tuple, int]]] = defaultdict(lambda: defaultdict(lambda: {}))
        self._min_path_lengths: dict[str, dict[str, int]] = {}
        self._forward_ordering = nx.MultiDiGraph()
        self._reverse_ordering = nx.MultiDiGraph()
        self._forward_ordering.clear()
        self._reverse_ordering.clear()

        # make topology graph undirected to apply the st-numbering algorithm
        # all pair of links (A-B, B-A) will be treated as a single undirected link
        undirected_topo = nx.Graph(topology)

        # use dfs to convert the graph into directed acyclic graph and this graph will be used by all nodes
        # how to choose source is a good question
        # for now we will take the node that has the longest path in the graph
        source_node = self._node_with_longest_path(undirected_topo)
        numbered_topo = self._dag_convert(undirected_topo, source_node)

        # using this dag, create two directed acyclic graphs from original topology
        # first graph is created according to dfs numbers
        # second graph is the first graph but with all edges' direction reverted (dfs are reverted accordingly)
        # all edges of the original topology are present either in first or in second graph
        max_number = len(numbered_topo.nodes) + 1
        for node_id, node_data in numbered_topo.nodes(data=True):
            # add node into forward ordering
            self._forward_ordering.add_node(node_id, **node_data)

            # add same node with reversed number into reverse ordering
            node_data['dfs_number'] = max_number - node_data['dfs_number']
            self._reverse_ordering.add_node(node_id, **node_data)

            # add links
            for _, neighbor_id in numbered_topo.edges(nbunch=node_id):
                # numbered_topo is a DiGraph without multi links
                # we need to look into original topology to get multi links

                # for each link (A, B) from numbered_topo, get all links (A, B, x) from original topology
                # and add them to forward ordering
                for edge_index, edge_data in self._get_topology_edges_between_nodes(topology, node_id, neighbor_id):
                    self._forward_ordering.add_edge(node_id, neighbor_id, key=edge_index, **edge_data)

                # for each link (A, B) from numbered_topo, get all links (B, A, x) from original topology
                # and add them to reverse ordering
                for edge_index, edge_data in self._get_topology_edges_between_nodes(topology, neighbor_id, node_id):
                    self._reverse_ordering.add_edge(neighbor_id, node_id, key=edge_index, **edge_data)

        self._calculate_path_lengths(self._forward_ordering)
        self._calculate_path_lengths(self._reverse_ordering)
        self._calculate_min_path_lengths(topology)
        if self.gml not in self.gml_dict:
            self.gml_dict[self.gml] = self._forward_ordering, self._reverse_ordering, self._forward_path_lengths, self._reverse_path_lengths, self._min_path_lengths, self.hash_paths

    def _dag_convert(self, graph: nx.Graph, s_node: str) -> nx.DiGraph:
        # convert graph into directed acyclic graph using dfs search
        graph_nodes = graph.nodes(data=True)
        current_number = 1
        graph_nodes[s_node]['dfs_number'] = current_number
        for node_from, node_to in dfs_edges(graph, s_node):
            current_number += 1
            graph_nodes[node_to]['dfs_number'] = current_number

        dag_graph = nx.DiGraph()
        dag_graph.add_nodes_from(graph_nodes)
        for node_from, node_to in graph.edges():
            if graph_nodes[node_from]['dfs_number'] < graph_nodes[node_to]['dfs_number']:
                dag_graph.add_edge(node_from, node_to)
            else:
                dag_graph.add_edge(node_to, node_from)

        return dag_graph

    @contextmanager
    def _hide_nodes(self, graph: nx.MultiDiGraph, nodes: Iterable[str]) -> None:
        nodes_data = {}
        edges = []
        for node in nodes:
            try:
                nodes_data[node] = graph.nodes[node]
            except KeyError:
                continue
            edges.extend(list(graph.out_edges(nbunch=node, data=True, keys=True)))
            edges.extend(list(graph.in_edges(nbunch=node, data=True, keys=True)))
            graph.remove_node(node)

        if graph is self._forward_ordering:
            self._forward_removed_nodes.update(nodes)
        else:
            self._reverse_removed_nodes.update(nodes)

        try:
            yield
        finally:
            for node, node_data in nodes_data.items():
                graph.add_node(node, **node_data)
            graph.add_edges_from(edges)

            if graph is self._forward_ordering:
                self._forward_removed_nodes.difference_update(nodes)
            else:
                self._reverse_removed_nodes.difference_update(nodes)

    @contextmanager
    def _ignore_nodes(self, graph: nx.MultiDiGraph, nodes: Iterable[str]):
        self.get_removed_nodes(graph).update(nodes)
        try:
            yield
        finally:
            self.get_removed_nodes(graph).difference_update(nodes)

    def _find_possible_nexthops(self, topology: nx.MultiDiGraph,
                                source: str, destination: str, original_length: int, curr_len: int = 0) -> list[GraphPathElement]:
        """
        Get next hops of source on simple paths to destination
            original_length and self.length_cutoff is used to limit the hop length of the possible path
        :param topology: topology graph
        :param source: source node
        :param destination: destination node
        :param original_length: length of the shortest hop path from source to destination
        :return: list of nexthops
        """
        possible_nexthops = []
        src_out_edges = topology.edges(nbunch=source, keys=True)
        removed_nodes = self.get_removed_nodes(topology)
        for _, neighbor, edge_index in src_out_edges:
            if neighbor in removed_nodes:
                continue
            try:
                removed_tuple = tuple(sorted(removed_nodes))
                length = self.get_lengths(topology)[neighbor][destination][removed_tuple]
            except KeyError:
                continue
            if curr_len + length + 1 <= int(original_length * self._length_cutoff):
                possible_nexthops.append(GraphPathElement(from_=source, to_=neighbor, index=edge_index))
        return possible_nexthops

    def _check_change_direction_path_possible(
                self, forward_graph: nx.MultiDiGraph, reverse_graph: nx.MultiDiGraph,
                source: str, destination: str, original_length: int, curr_len: int = 0) -> bool:
        """
        Check if vertex exists such that:
        1. it is reachable from source in forward_graph
        2. there is a simple path from that vertex to destination in reverse_graph
        """
        fwd_tuple = self.get_removed_tuple(forward_graph)
        rev_tuple = self.get_removed_tuple(reverse_graph)
        for _, node_to in dfs_edges_ignore(forward_graph, source, ignore=self.get_removed_nodes(forward_graph)):
            if node_to == destination:
                continue
            try:        
                depth = self.get_lengths(forward_graph)[source][node_to][fwd_tuple]
                length = self.get_lengths(reverse_graph)[node_to][destination][rev_tuple]
            except KeyError:
                continue
            if curr_len + length + depth + 1 <= int(original_length * self._length_cutoff):
                return True

    def _find_nexthops_with_change_direction(
                self, forward_graph: nx.MultiDiGraph, reverse_graph: nx.MultiDiGraph,
                source: str, destination: str, original_length: int, curr_len: int = 0) -> list[GraphPathElement]:
        """
        Search for possible nexthops by allowing to change the direction once
        Use dfs search in forward_graph looking for a node that has a simple path in reverse_graph
            If there is, here is our nexthop.

        :param forward_graph: initial graph to look through
        :param reverse_graph: graph to look through after changing direction
        :param source: source node
        :param destination: destination node
        :return: list of nexthops
        """
        possible_nexthops = []
        with self._ignore_nodes(reverse_graph, [source]):
            removed_tuple = self.get_removed_tuple(reverse_graph)
            src_out_edges = forward_graph.edges(nbunch=source, keys=True)
            for _, neighbor, edge_index in src_out_edges:
                if neighbor in removed_tuple:
                    continue
                try:
                    length = self.get_lengths(reverse_graph)[neighbor][destination][removed_tuple]
                    if curr_len + length + 1 <= int(original_length * self._length_cutoff):
                        possible_nexthops.append(GraphPathElement(from_=source, to_=neighbor, index=edge_index))
                except KeyError:
                    # no direct path => try to change direction in some vertex
                    if self._check_change_direction_path_possible(
                                forward_graph, reverse_graph, neighbor, destination, original_length, curr_len=curr_len + 1):
                        possible_nexthops.append(GraphPathElement(from_=source, to_=neighbor, index=edge_index))

        return possible_nexthops
    
    def get_removed_nodes(self, g: nx.MultiDiGraph):
        if g is self._forward_ordering:
            return self._forward_removed_nodes
        elif g is self._reverse_ordering:
            return self._reverse_removed_nodes
        else:
            raise ValueError("g must be either forward or reverse ordering")
        
    def get_removed_tuple(self, g: nx.MultiDiGraph):
        if g is self._forward_ordering:
            return tuple(sorted(self._forward_removed_nodes))
        elif g is self._reverse_ordering:
            return tuple(sorted(self._reverse_removed_nodes))
        else:
            raise ValueError("g must be either forward or reverse ordering")
        
    def get_lengths(self, g: nx.MultiDiGraph):
        if g is self._forward_ordering:
            return self._forward_path_lengths
        elif g is self._reverse_ordering:
            return self._reverse_path_lengths
        else:
            raise ValueError("g must be either forward or reverse ordering")

    def _calculate_from_previous(self, topology: nx.MultiDiGraph, source: str,
                                 previous: Optional[str], destination: str, start_node: str, curr_len: int = 0) -> list[GraphPathElement]:
        # in our orderings only paths that exist are the ones that go from:
        #   lower dfs_number to higher for forward ordering,
        #   higher dfs_number to lower for reverse ordering
        # so we must use the correct graph depending on source and destination
        max_len = topology.number_of_nodes() + 1
        min_len = self._min_path_lengths[start_node][destination]
        try:
            if (self._forward_ordering.nodes[source]['dfs_number'] >
                    self._forward_ordering.nodes[previous]['dfs_number']):
                forward_graph = self._forward_ordering
                reverse_graph = self._reverse_ordering
            else:
                forward_graph = self._reverse_ordering
                reverse_graph = self._forward_ordering
        except KeyError:
            # source or destination was removed from topology
            raise nx.NodeNotFound

        with self._ignore_nodes(forward_graph, [previous, start_node]):
            with self._ignore_nodes(reverse_graph, [previous, start_node]):
                # check next hops in forward graph
                try:
                    removed_tuple = self.get_removed_tuple(forward_graph)
                    original_length = self.get_lengths(forward_graph)[source][destination][removed_tuple]
                    return self._find_possible_nexthops(forward_graph, source, destination, min_len, curr_len=curr_len)
                except KeyError:
                    pass

                # check next hops in reverse graph
                try:
                    removed_tuple = self.get_removed_tuple(reverse_graph)
                    original_length = self.get_lengths(reverse_graph)[source][destination][removed_tuple]
                    return self._find_possible_nexthops(reverse_graph, source, destination, min_len, curr_len=curr_len)
                except KeyError:
                    pass

                # If there are no paths in forward of reverse graphs, find node such that:
                # 1. it is reachable from current source in forward graph
                # 2. destination is reachable from it in reverse graph
                # NOTE: should not we try reverse direction?
                result = self._find_nexthops_with_change_direction(
                        forward_graph, reverse_graph, source, destination, min_len, curr_len=curr_len)
                if not result:
                    raise nx.NetworkXNoPath
                return result

    def _calculate(self, topology: nx.MultiDiGraph, source: str,
                  previous: Optional[str], destination: str, start_node: str, curr_len: int = 0) -> list[GraphPathElement]:
        if previous is not None:
            return self._calculate_from_previous(topology, source, previous, destination, start_node, curr_len)
        max_len = topology.number_of_nodes() + 1

        # previous is none so this is the first node that sees the current flow
        # here we can provide more nexthop options than usual
        # NOTE: should not we check that source is start_node? start_node is not removed

        possible_nexthops = []

        min_length = self._min_path_lengths[start_node][destination]

        # find next hops in forward ordering
        try:
            original_length_forward = self._forward_path_lengths[source][destination][tuple()]
            possible_nexthops.extend(
                self._find_possible_nexthops(self._forward_ordering, source, destination, min_length, curr_len=curr_len)
            )
        except KeyError:
            original_length_forward = max_len

        # find next hops in reverse ordering
        try:
            original_length_reverse = self._reverse_path_lengths[source][destination][tuple()]
            
            possible_nexthops.extend(
                self._find_possible_nexthops(self._reverse_ordering, source, destination, min_length, curr_len=curr_len)
            )
        except KeyError:
            original_length_reverse = max_len

        # find next hops allowing 1 direction change in path
        original_length = min(original_length_reverse, original_length_forward)
        possible_nexthops.extend(
            self._find_nexthops_with_change_direction(
                self._forward_ordering, self._reverse_ordering, source, destination, min_length, curr_len=curr_len
            )
        )
        possible_nexthops.extend(
            self._find_nexthops_with_change_direction(
                self._reverse_ordering, self._forward_ordering, source, destination, min_length, curr_len=curr_len
            )
        )

        if self._store_nexthops:
            self._flow_set_nexthops.append(len(possible_nexthops))
        if not possible_nexthops:
            raise nx.NetworkXNoPath
        return possible_nexthops
