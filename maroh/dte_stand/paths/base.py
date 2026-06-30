import abc
import networkx
from typing import Optional
from dte_stand.data_structures import GraphPathElement
from dte_stand.config import Config


class BasePathCalculator:
    def __init__(self):
        self._number_of_nexthops: list[list[list[int]]] = []
        self._episode_nexthops: list[list[int]] = []
        self._flow_set_nexthops: list[int] = []
        config = Config.config()
        self._store_nexthops = config.store_nexthops
        self.hash_paths = {}
        self.gml = None

    def end_iteration(self):
        if not self._episode_nexthops:
            return
        self._number_of_nexthops.append(self._episode_nexthops)
        self._episode_nexthops = []

    def end_flow_set(self):
        if not self._flow_set_nexthops:
            return
        self._episode_nexthops.append(self._flow_set_nexthops)
        self._flow_set_nexthops = []

    @property
    def number_of_nexthops(self):
        return self._number_of_nexthops

    def prepare_iteration(self, topology: networkx.MultiDiGraph) -> None:
        ...

    def calculate(self, topology: networkx.MultiDiGraph, source: str,
                  previous: Optional[str], destination: str, start_node: str, curr_len: int = 0) -> list[GraphPathElement]:
        if (self.gml, source, previous, destination, start_node) in self.hash_paths:
            return self.hash_paths[(self.gml, source, previous, destination, start_node)]
        else:
            res = self._calculate(topology, source, previous, destination, start_node, curr_len=curr_len)
            self.hash_paths[(self.gml, source, previous, destination, start_node)] = res
            return res

    @abc.abstractmethod
    def _calculate(self, topology: networkx.MultiDiGraph, source: str,
                  previous: Optional[str], destination: str, start_node: str, curr_len: int = 0) -> list[GraphPathElement]:
        ...
