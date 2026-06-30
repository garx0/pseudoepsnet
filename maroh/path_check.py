import torch
import sys
import time
import networkx as nx
from collections import deque
from dte_stand.data_structures import HashWeights
from dte_stand.hash_function import RandomHashFunction2
from dte_stand.paths.dag_calculator import DAGCalculator
from dte_stand.data_structures import InputData
from dte_stand.config import Config
import os


topo = nx.MultiDiGraph(nx.read_gml(os.path.join(sys.argv[1], 'topology.gml')))
Config.load_config(os.path.join(sys.argv[1], '../..'))

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
# flow_sd = []

path_calc = DAGCalculator()
path_calc.prepare_iteration(topo)

flow_hashes_info = {}
flow_lists = {}
all_paths = []
path_idx = 0
cnt = 0

# for start, end, key, data in topo.edges(keys=True, data=True):
#     print(start, end, key, data)

print('path calculator initialized')

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
                    all_paths.append({'source':start, 'destination':end, 'edges':curr_edge_path + [way]})
                else:
                    q1.append(curr_node_path + [way.to_])
                    q2.append(curr_edge_path + [way])
                

# for path in all_paths:
#     length = len(path['edges'])
#     end = path['edges'][-1].to_
#     start = path['edges'][0].from_
#     if end != path['destination'] or start != path['source']:
#         print("ERROR in path:", path)
#     if length > nx.shortest_path_length(topo, start, end) * 2:
#         print("ERROR: path too long:", path)
#         pass

for start in topo.nodes():
    for end in topo.nodes():
        if start == end:
            continue
        paths_se = [p for p in all_paths if p['source'] == start and p['destination'] == end]
        if len(paths_se) == 0:
            print("ERROR: no path found for", start, end)

print(f"Total DAG paths found: {len(all_paths)}")

path_edges = {} # dict{(source, dest): dict{node: list(node, node, ...)}}

num_of_splits = 0

nexthops_dict = {} # dict{(source, dest): [nexthop]}
nexthops_prev_dict = {} # dict{(prev, source, dest): [nexthop]}
num_of_nexthops_prev = 0
num_of_nexthops_splits = 0
num_of_nexthops = 0

for path in all_paths:
    start = path['edges'][0].from_
    end = path['edges'][-1].to_

    if (start, end) not in path_edges:
        path_edges[(start, end)] = {}
    for i in range(len(path['edges'])):
        edge = path['edges'][i]
        prev = path['edges'][i-1].from_ if i > 0 else None
        node = edge.from_

        if (prev, node, end) not in nexthops_prev_dict:
            nexthops_prev_dict[(prev, node, end)] = [edge.to_]
            num_of_nexthops_prev += 1
        elif edge.to_ not in nexthops_prev_dict[(prev, node, end)]:
            nexthops_prev_dict[(prev, node, end)].append(edge.to_)
            num_of_nexthops_prev += 1

        if(node, end) not in nexthops_dict:
            nexthops_dict[(node, end)] = [edge.to_]
            num_of_nexthops += 1
        elif edge.to_ not in nexthops_dict[(node, end)]:
            if len(nexthops_dict[(node, end)]) == 1:
                num_of_nexthops_splits += 1
            nexthops_dict[(node, end)].append(edge.to_)
            num_of_nexthops_splits += 1
            num_of_nexthops += 1
        

        if node not in path_edges[(start, end)]:
            path_edges[(start, end)][node] = [edge.to_]
        elif edge.to_ not in path_edges[(start, end)][node]:
            if len(path_edges[(start, end)][node]) == 1:
                num_of_splits += 1
            path_edges[(start, end)][node].append(edge.to_)
            num_of_splits += 1

print('num_of_splits =', num_of_splits)
print('next hops =', num_of_nexthops)
print('next hops splits =', num_of_nexthops_splits)
print('next hops with prev =', num_of_nexthops_prev)


# KSP paths

K = 4

all_paths_ksp = []

topo2 = nx.DiGraph(topo)

for start in topo2.nodes():
    for end in topo2.nodes():
        if start == end:
            continue
        shortest_len = nx.shortest_path_length(topo2, start, end)
        max_len = shortest_len * 2


        paths = []

        for path in nx.shortest_simple_paths(topo2, start, end):
            length = len(path) - 1

            if length > max_len:
                break

            paths.append((path, length))

            if len(paths) >= 4:
                break
        all_paths_ksp.extend([{'source': start, 'destination': end, 'nodes': p[0]} for p in paths])

print('Total KSP paths:', len(all_paths_ksp))

ksp_splits = 0
ksp_path_edges = {}

nexthops_dict = {} # dict{(source, dest): [nexthop]}
num_of_nexthops_splits = 0
num_of_nexthops = 0

for path in all_paths_ksp:
    start = path['source']
    end = path['destination']
    if (start, end) not in ksp_path_edges:
        ksp_path_edges[(start, end)] = {}
    for i in range(len(path['nodes'])):
        node = path['nodes'][i]

        if i != len(path['nodes']) - 1 and (node, end) not in nexthops_dict:
            nexthops_dict[(node, end)] = [path['nodes'][i+1]]
            num_of_nexthops += 1
        elif i != len(path['nodes']) - 1 and path['nodes'][i+1] not in nexthops_dict[(node, end)]:
            nexthops_dict[(node, end)].append(path['nodes'][i+1])
            if len(nexthops_dict[(node, end)]) == 2:
                num_of_nexthops_splits += 1
            num_of_nexthops_splits += 1
            num_of_nexthops += 1

        if i != len(path['nodes']) - 1 and node not in ksp_path_edges[(start, end)]:
            ksp_path_edges[(start, end)][node] = [path['nodes'][i+1]]
        elif i != len(path['nodes']) - 1 and path['nodes'][i+1] not in ksp_path_edges[(start, end)][node]:
            if ksp_path_edges[(start, end)][node].__len__() == 1:
                ksp_splits += 1
            ksp_path_edges[(start, end)][node].append(path['nodes'][i+1])
            ksp_splits += 1

print('ksp splits =', ksp_splits)
print('ksp next hops =', num_of_nexthops)
print('ksp next hops slits =', num_of_nexthops_splits)