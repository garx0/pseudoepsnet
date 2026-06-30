from collections import deque
import cvxpy as cp
import networkx as nx
# from itertools import islice
from dte_stand.data_structures.flows import Flows
from dte_stand.data_structures.topology import Topology
import numpy as np

def load_topo(path, node_map=None, path_to_graph_changes=None, tim=0):
    # topo = nx.MultiDiGraph(nx.read_gml(path))
    topo_obj = Topology(path, path_to_graph_changes=path_to_graph_changes)
    topo, _ = topo_obj.get(tim)

    idx = 0
    for i, j, m in topo.edges:
        topo[i][j][m]['label'] = topo[i][j][m]['id']
        topo[i][j][m]['id'] = idx
        idx += 1

    if node_map is None:
        idx = 0
        node_map = {}
        for node in topo.nodes:
            node_map[node] = idx
            idx += 1

    nx.relabel_nodes(topo, node_map, copy=False)
    return topo, node_map

def convert_topo(topo, node_map=None):
    'topo: nx graph loaded using Topology class and get method. use if loaded topology other way than load_topo function'
    topo2 = topo.copy()
    idx = 0
    for i, j, m in topo2.edges:
        topo2[i][j][m]['label'] = topo2[i][j][m]['id']
        topo2[i][j][m]['id'] = idx
        idx += 1

    if node_map is None:
        idx = 0
        node_map = {}
        for node in topo2.nodes:
            node_map[node] = idx
            idx += 1

    nx.relabel_nodes(topo2, node_map, copy=False)
    return topo2, node_map

def load_flows(path):
    flows_obj = Flows(path)
    return flows_obj

def get_flows_at(flows_obj, tim, node_map, numbers=True):
    flows_obj_t = flows_obj.get(tim)
    flows_dicts = []
    for flow in flows_obj_t:
        if flow.start in node_map and flow.end in node_map:
            flows_dicts.append({
                'start': node_map[flow.start] if numbers else flow.start,
                'end': node_map[flow.end] if numbers else flow.end,
                'rate': flow.bandwidth,
                'hash': flow.hash
                })
    return flows_dicts

class LPSolver:
    def __init__(self, topo, node_map, path_calc):
        self.topo, self.node_map = convert_topo(topo)
        # self.node_map = node_map
        self.path_calc = path_calc
        # flow_hashes_info = {}
        # flow_lists = {}
        all_paths = {}
        # path_idx = 0
        cnt = 0

        for start in topo.nodes():
            for end in topo.nodes():
                if start == end:
                    continue
                q1 = deque([[start]])
                q2 = deque([[]])
                while q1:
                    curr_node_path = q1.popleft()
                    curr_edge_path = q2.popleft()
                    curr = curr_node_path[-1]
                    prev = curr_node_path[-2] if len(curr_node_path) > 1 else None
                    try:
                        # print(f"{cnt}: {curr=}, {prev=}, {end=}, {start=}")
                        cnt += 1
                        ways = list(self.path_calc.calculate(topo, curr, prev, end, start, len(curr_edge_path)))
                    except:
                        ways = []
                    nexthop_tuples = []
                    for way in ways:
                        nexthop_tuple = (way.from_, way.to_, way.index)
                        if nexthop_tuple in nexthop_tuples:
                            continue
                        nexthop_tuples.append(nexthop_tuple)
                        if way.to_ == end:
                            if (start, end) not in all_paths:
                                all_paths[(start, end)] = [curr_edge_path + [way]]
                            else:
                                all_paths[(start, end)].append(curr_edge_path + [way])

                        else:
                            q1.append(curr_node_path + [way.to_])
                            q2.append(curr_edge_path + [way])


        self.paths = {}
        for k in all_paths:
            key = (self.node_map[k[0]], self.node_map[k[1]])
            self.paths[key] = []
            for path in all_paths[k]:
                node_ids = []
                for way in path:
                    node_ids.append(self.node_map[way.from_])
                node_ids.append(self.node_map[way.to_])
                self.paths[key].append(node_ids)
    
    def export_solution(self, flows, path_vars):
        node_names = [None] * len(self.node_map)
        for name in self.node_map:
            node_names[self.node_map[name]] = name
        paths_and_splits = {}
        for i in range(len(flows)):
            flow = flows[i]
            start, end = node_names[flow['start']], node_names[flow['end']]
            paths_arr = []
            for path in self.paths[(flow['start'], flow['end'])]:
                names_path = []
                for node in path:
                    names_path.append(node_names[node])
                paths_arr.append(names_path)
            paths_and_splits[(start, end)] = (paths_arr, path_vars[i].value.tolist())
        return paths_and_splits

    def solve(self, flows, verbose=False, save=False):
        path_vars = []
        for flow in flows:
            path_vars.append(cp.Variable(len(self.paths[(flow['start'], flow['end'])]), nonneg=True))

        z = cp.Variable()

        constraints = []
        for i, flow in enumerate(flows):
            constraints.append(cp.sum(path_vars[i]) == flow['rate'])

        edge_list = list(self.topo.edges())
        for u, v in edge_list:
            edge_cap = self.topo[u][v][0]['bandwidth']
            load_on_edge = 0

            for i, flow in enumerate(flows):
                for j, path in enumerate(self.paths[(flow['start'], flow['end'])]):
                    path_edges = list(zip(path, path[1:]))
                    if (u, v) in path_edges:
                        load_on_edge += path_vars[i][j]

            constraints.append(load_on_edge / edge_cap <= z)

        problem = cp.Problem(cp.Minimize(z), constraints)
        problem.solve(solver=cp.SCIPY, verbose=verbose)

        # for i, flow in enumerate(flows):
        #     print(f"Flow {i} splits: {path_vars[i].value.round(2)}")



        return z.value, self.export_solution(flows, path_vars) if save else None
