from dte_stand.hash_function import RandomHashFunction2
from dte_stand.data_structures import HashWeights, InputData, Bucket
from dte_stand.data_structures.paths import GraphPathElement

def test_flow_hash(solution, topo, flows):
    max_mlu = 0.0
    for _, _, edge_data in topo.edges(data=True):
        edge_data['current_bandwidth'] = 0
    for flow in flows:
        start, end = flow['start'], flow['end']
        paths, splits = solution[(start, end)]
        path_buckets = [Bucket(edge=i, weight=splits[i]) for i in range(len(splits))]
        bucket = RandomHashFunction2._choose_nexthop(None, path_buckets, 0, hash=flow['hash'])
        path = paths[bucket]
        for i in range(len(path)-1):
            topo.edges[path[i], path[i+1], 0]['current_bandwidth'] += flow['rate']
    for i, j in topo.edges():
        util = topo[i][j][0]['current_bandwidth'] / topo[i][j][0]['bandwidth']
        max_mlu = max(max_mlu, util)
    return max_mlu