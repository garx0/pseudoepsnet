from collections import defaultdict
from typing import Optional
from dataclasses import dataclass
from dte_stand.data_structures.paths import GraphPathElement


@dataclass
class Bucket:
    """
    Bucket represents and edge in the graph and its hash weight
    Choosing a bucket means choosing its edge (GraphPathElement) as the nexthop in path
    """
    edge: GraphPathElement
    weight: float


def sort_buckets(b: Bucket):
    """
    Function to use in python sort to sort buckets
    Buckets are sorted by comparing its destination node's name as a string
    """
    return b.edge.to_


class HashWeights:
    """
    Class to store hash weights produced by the algorithm

    Structure:
    weights are different for each pair of source and destination node. For each such pair a dict is stored,
        which contains resulted hash weights for each edge.
    Edge is the key of this dict - it goes from "start_node" to "neighbor_node" and has index "edge_index".
        Edge index is necessary because we are considering a MultiDiGraph
        (edges can be duplicated and index is used to differentiate them)
    The value of the dict is the weight of the edge
    """
    def __init__(self):
        self._weights: defaultdict[tuple[str, str], dict[tuple[str, int], Bucket]] = defaultdict(lambda: {})

    def put(self, start_node: str, end_node: str, neighbor_node: str, edge_index: int, weight: float) -> None:
        bucket = Bucket(
                edge=GraphPathElement(
                        from_=start_node,
                        to_=neighbor_node,
                        index=edge_index
                ),
                weight=weight
        )
        self._weights[(start_node, end_node)].update({(neighbor_node, edge_index): bucket})

    def get_weight(self, start_node: str, end_node: str, neighbor_node: str, edge_index: int) -> Optional[int]:
        try:
            return self._weights.get((start_node, end_node), {}).get((neighbor_node, edge_index)).weight
        except AttributeError:
            # no such bucket
            return None

    def get_bucket_list(self, start_node: str, end_node: str) -> list[Bucket]:
        """
        returns hash weights for a given pair of source and destination nodes, stored as buckets
        buckets are sorted in an arbitrary way, for the sake of consistency
        """
        return sorted(self._weights.get((start_node, end_node), {}).values(), key=sort_buckets)
