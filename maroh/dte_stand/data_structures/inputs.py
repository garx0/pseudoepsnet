import os
from dte_stand.data_structures.topology import Topology
from dte_stand.data_structures.flows import Flows


class InputData:
    # Inference experiment input data (topo and flows)
    def __init__(self, path_to_folder, flows_name="flows.json", ignore_topology_changes=False):
        self._flows = Flows(os.path.join(path_to_folder, flows_name))
        self._topology = Topology(os.path.join(path_to_folder, 'topology.gml'),
                                  os.path.join(path_to_folder, 'topology_changes.json')
                                  if not ignore_topology_changes else None)

    @property
    def topology(self):
        return self._topology

    @property
    def flows(self):
        return self._flows
