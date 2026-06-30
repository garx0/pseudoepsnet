from collections import deque
from dte_stand.paths.dag_calculator import DAGCalculator
from dte_stand.config import Config
import os
import sys
import cvxpy as cp
import networkx as nx
import numpy as np
import json
from itertools import islice
import os

d = f'./opt_test/{sys.argv[1]}'

topo = nx.MultiDiGraph(nx.read_gml(os.path.join(d, 'topology.gml')))
Config.load_config(d)

idx = 0
for i, j, m in topo.edges:
    topo[i][j][m]['label'] = topo[i][j][m]['id']
    topo[i][j][m]['id'] = idx
    idx += 1

idx = 0
node_map = {}
for node in topo.nodes:
    node_map[node] = idx
    idx += 1

nx.relabel_nodes(topo, node_map, copy=False)

path_calc = DAGCalculator()
path_calc.prepare_iteration(topo)

flow_hashes_info = {}
flow_lists = {}
all_paths = {}
path_idx = 0
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
                ways = list(path_calc.calculate(topo, curr, prev, end, start, len(curr_edge_path)))
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


paths = {}
for k in all_paths:
    paths[k] = []
    for path in all_paths[k]:
        node_ids = []
        for way in path:
            node_ids.append(way.from_)
        node_ids.append(way.to_)
        paths[k].append(node_ids)

j = json.load(open(os.path.join(d, f'flows{sys.argv[2] if len(sys.argv) > 2 else ""}.json'), 'r'))
flows = j['flows']
print(f"Number of flows: {len(flows)}")
for i in range(len(flows)):
    flows[i]['rate'] = flows[i]['all_bandwidth']['0']
    flows[i]['start'] = node_map[flows[i]['start']]
    flows[i]['end'] = node_map[flows[i]['end']]

path_vars = []
for flow in flows:
    path_vars.append(cp.Variable(len(paths[(flow['start'], flow['end'])]), nonneg=True))

z = cp.Variable()

constraints = []
for i, flow in enumerate(flows):
    constraints.append(cp.sum(path_vars[i]) == flow['rate'])

edge_list = list(topo.edges())
for u, v in edge_list:
    edge_cap = topo[u][v][0]['bandwidth']
    load_on_edge = 0
    
    for i, flow in enumerate(flows):
        for j, path in enumerate(paths[(flow['start'], flow['end'])]):
            path_edges = list(zip(path, path[1:]))
            if (u, v) in path_edges:
                load_on_edge += path_vars[i][j]
    
    constraints.append(load_on_edge / edge_cap <= z)

problem = cp.Problem(cp.Minimize(z), constraints)
problem.solve(solver=cp.SCIPY, verbose=False)

print(f"Optimal MLU: {z.value}")
# for i, flow in enumerate(flows):
#     print(f"Flow {i} splits: {path_vars[i].value.round(2)}")