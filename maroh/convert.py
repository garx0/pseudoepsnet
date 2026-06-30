import sys
import networkx

if __name__ == '__main__':
    try:
        topo_filepath = sys.argv[1]

        result_filepath = sys.argv[2]
    except IndexError:
        print('Usage: convert.py path_to_topo_file path_to_result_file\n'
              'path_to_topo_file is a path to topology in gml format to be converted for dte stand.\n'
              'path_to_result_file is a path where to save the converted topology\n')
        exit(1)

    with open(topo_filepath, mode='rb') as f:
        original = networkx.readwrite.read_gml(f)

    original_no_edges = networkx.create_empty_copy(original)
    converted = networkx.MultiDiGraph(original_no_edges)

    for node_from, node_to, edge_dict in original.edges(data=True):
        if 'bandwidth' not in edge_dict:
            edge_dict['bandwidth'] = 0
        edge_dict['current_bandwidth'] = 0
        converted.add_edge(node_from, node_to, **edge_dict)
        if 'id' in edge_dict:
            edge_dict['id'] += '_r'
        converted.add_edge(node_to, node_from, **edge_dict)

    with open(result_filepath, mode='wb') as f:
        networkx.readwrite.write_gml(converted, f)
