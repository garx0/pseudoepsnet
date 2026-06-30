import json
import networkx
from typing import Optional, Dict, Any
from pydantic import BaseModel, ConfigDict, RootModel
import copy
from time import sleep

N_IO_TRIES = 12

class MissingElements(BaseModel):
    missing_nodes: list[str]
    missing_links: list[tuple[str, str, int]]


class TopologyChanges(RootModel):
    root: Dict[str, MissingElements]

    def __iter__(self):
        return iter(self.root)

    def __getitem__(self, item):
        return self.root[item]

    def keys(self):
        return self.root.keys()


class Topology:
    def __init__(self, path_to_graph: str, path_to_graph_changes: Optional[str]):
        initial_topology = None
        for _ in range(N_IO_TRIES):
            try:
                with open(path_to_graph, 'rb') as file_graph:
                    initial_topology = networkx.readwrite.read_gml(file_graph)
                break
            except Exception as e:
                print(f"IOERROR: {e}")
                sleep(5)
        if initial_topology is None:
            raise Exception(f"wasn't able to read {path_to_graph} due to repeated IOERROR")

        self._initial_topology: networkx.MultiDiGraph = initial_topology

        if path_to_graph_changes is not None:
            topology_changes_data = None
            for _ in range(N_IO_TRIES):
                try:
                    with open(path_to_graph_changes, 'r') as f:
                        topology_changes_data = json.load(f)
                    break
                except Exception as e:
                    print(f"IOERROR: {e}")
                    sleep(5)
            if topology_changes_data is None:
                raise Exception(f"wasn't able to read {path_to_graph_changes} due to repeated IOERROR")

            self._topology_changes = TopologyChanges.model_validate(topology_changes_data)
        else:
            self._topology_changes = {}
        self._changed_at: list[int] = [int(x) for x in self._topology_changes.keys()]
        self._changed_at.sort()
        # remember the last time get was called. Used for determining the change to topology to apply
        self._previous_time = -1

    def get(self, current_time: int) -> tuple[networkx.MultiDiGraph, Optional[int]]:
        current_topology: networkx.MultiDiGraph = copy.deepcopy(self._initial_topology)

        # get time of latest change. It is the first point of change between previous time and current time
        try:
            latest_change = [t for t in self._changed_at
                             if self._previous_time < t <= current_time][0]
        except IndexError:
            # no changes in topology between previous and current
            # so take the last change before previous time
            try:
                latest_change = [t for t in self._changed_at if t <= self._previous_time][-1]
            except IndexError:
                # no changes at all
                return current_topology, None
        self._previous_time = int(latest_change)

        # apply change
        try:
            current_changes = self._topology_changes[str(latest_change)]
        except KeyError:
            # no changes in topology at given time slot
            return current_topology, None

        for node_id in current_changes.missing_nodes:
            current_topology.remove_node(node_id)
        for node1_id, node2_id, index in current_changes.missing_links:
            current_topology.remove_edge(node1_id, node2_id, index)
        return current_topology, int(latest_change)
