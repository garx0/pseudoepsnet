import networkx
from typing import Optional
from networkx.algorithms.shortest_paths import shortest_path_length
from networkx.algorithms.simple_paths import all_simple_edge_paths
from dte_stand.paths.base import BasePathCalculator
from dte_stand.data_structures import GraphPathElement


class DummyPathCalculator(BasePathCalculator):
    """
    Calculator that returns only shortest paths by hops
    This ensures there will be no cycles but often produces only one path
    """
    def calculate(self, topology: networkx.MultiDiGraph, source: str,
                  previous: Optional[str], destination: str, start_node: str) -> list[GraphPathElement]:
        length = shortest_path_length(topology, source, destination)
        nx_paths = all_simple_edge_paths(topology, source, destination, cutoff=length)
        nexthops: list[GraphPathElement] = []
        for path in nx_paths:
            from_, to_, index = path[0]
            nexthops.append(GraphPathElement(from_=from_, to_=to_, index=index))
        return nexthops

