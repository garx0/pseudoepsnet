from dte_stand.paths.dag_calculator import DAGCalculator
from dte_stand.config import Config
import os
import numpy as np
import argparse
from datetime import datetime
import pickle

from generator.lp_solver import load_topo, load_flows, get_flows_at, LPSolver

def parse_args():
    parser = argparse.ArgumentParser(description='Optimal MLU Calculation')

    parser.add_argument("dir", type=str,
        help="path to directory with topology.gml, flows.json")

    parser.add_argument("--iter", type=int, default=1,
        help="n_iterations, number of traffic matrices in flows.json")

    parser.add_argument('--flowsname', type=str, default='flows.json',
        help='Name of file with flows to process (default: flows.json)')
    
    parser.add_argument('--configpath', type=str, default='../../config.yaml',
        help='Name of file with flows to process (default: ../../config.yaml)')

    parser.add_argument('--save', action='store_true', help='Save solution')

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    exp_path = args.dir
    config_parent_path, config_filename = os.path.split(os.path.normpath(os.path.join(exp_path, args.configpath)))
    config_filename = os.path.splitext(config_filename)[0]
    Config.load_config(config_parent_path, modifier=config_filename[len("config"):])
    topo, node_map = load_topo(os.path.join(exp_path, 'topology.gml'))
    flows = load_flows(os.path.join(exp_path, args.flowsname))
    path_calc = DAGCalculator()
    path_calc.prepare_iteration(topo)
    dat_str = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    solver = LPSolver(topo, node_map, path_calc)
    
    n_iter = args.iter
    flowsperiod = 30000
    flowsname_noext = os.path.splitext(args.flowsname)[0]
    
    results = []
    for tim in range(0, n_iter * flowsperiod, flowsperiod):    
        flows_t = get_flows_at(flows, tim, node_map)
        print(f"time {tim}: Number of flows: {len(flows_t)}")
        value, solution = solver.solve(flows_t, save=args.save)
        print(f"time {tim}: Optimal MLU = {value}")
        filename_prefix = os.path.join(exp_path, f"{flowsname_noext}-opt-{tim}_{dat_str}")
        np.save(f'{filename_prefix}.npy', np.array([value]))
        if args.save:
            f = open(f'{filename_prefix}_sol.pkl', 'wb')
            pickle.dump(solution, f)
            f.close()
        results.append((tim, value))
    
    if len(results) > 1:
        print("\nResults:")
        for tim, value in results:
            print(f"time {tim:9}: Optimal MLU = {value}")