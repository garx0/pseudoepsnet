from generator.generator import UniformFlowGenerator
from generator.demand_generator import DemandMatrixGenerator
from generator.parsers import sndlib_brain as sndlib_brain_parser
from generator.weightmatrix_test import PhiTest
from generator.lp_solver import convert_topo, load_flows, get_flows_at, LPSolver
from generator.flow_hash_test import test_flow_hash
import json
import networkx
import os, sys
from datetime import datetime
import argparse
from pathlib import Path
import random
import numpy as np

from dte_stand.config import Config

from dte_stand.data_structures import HashWeights, InputData
from dte_stand.hash_function import RandomHashFunction2
from dte_stand.phi_calculator import PhiCalculator
from dte_stand.paths import DAGCalculator
from dte_stand.logger import init_logger

def parse_args():
    parser = argparse.ArgumentParser(description='Generate network flows')
    parser.add_argument('topology_path', type=Path,
                       help='path to .gml file with topology, or to directory containing topology.gml')
    parser.add_argument('--mode', choices=['gravity', 'standard'], default='gravity',
                       help='gravity (default) - gravity model, standard - algorithm of synthetic demand generation')
    parser.add_argument('--seed', type=int, default=None,
                       help='random seed for demand generation')
    parser.add_argument('--flows', type=int, default=2,
                       help='Max flow amount parameter. limits the maximum amount of flows that can exist between any pair of nodes at any time.'
                           'Actual amount of flows is a random number between this and the amount that carried from previous step'
                           'of the generation. So setting this number higher means there will be more flows generated but with'
                           'smaller bandwidths.')
    parser.add_argument('--intensity', type=float, default=0.4,
                        help='sqrt(max/min) of demand. Increasing this value increases overall demand.'
                        'Sets sqrt(min_bw_coef * max_bw_coef) for DemandMatrixGenerator arguments max_bw_coef, min_bw_coef)'
                        'Choose manually until achieving desired value of mean occupied bandwith (see "average bandwidth taken" in the end of console output).')
                        # TODO: make the selection automatic
    parser.add_argument('--bwvar', type=float, default=5,
                        help='Demand variation, sets max_bw_coef / min_bw_coef for DemandMatrixGenerator arguments max_bw_coef, min_bw_coef')
    parser.add_argument('--matr', type=int, default=1,
                       help='number of matrices in each flows file')

    parser.add_argument('--nfiles', type=int, default=100,
                       help='number of flows files')

    parser.add_argument('--out_path', type=Path, default=None,
                       help='json output file name (template, will be appended by number for each generated flows file)')

    parser.add_argument('--ecmp', action='store_true', help='Calculate ECMP distribution after generation')

    parser.add_argument('--solver', action='store_true', help='Calculate LPSolver result after generation')

    parser.add_argument('--changes', action='store_true', help='Consider topology changes in topology_changes.json when calculating ECMP and optimum')

    parser.add_argument('--interp', type=int, default=1, help='Generate --matr matrices and insert (interp - 1) matrices between each two '
                        'consecutive matrices via linear interpolation, resulting in (matr - 1) * interp + 1 matrices in total')

    parser.add_argument('--splitseq', action='store_true', help='In a sequence of matrices, generate individual file with flows for each matrix')

    return parser.parse_args()

def generate_from_dataset():
    # create generator and set its parameters
    generator = UniformFlowGenerator(5, 400000, 2000000)

    # parse folder that contains dataset files
    matrices = sndlib_brain_parser.parse_all('generator/dataset/')

    # run the generator
    result = generator.generate(matrices, 60000)

    # convert results to a string with optional pretty print - can be removed, only result.json() is needed
    # But results will be unreadable for a human is pretty print is removed
    pretty_res = json.dumps(json.loads(result.json()), indent=4)

    # open result file and write the flow data
    with open('flows.log', 'w') as f:
        f.write(pretty_res)


def generate_synthetic(topology_file_path: str,
                       out_file_path: str,
                       mode: str,
                       n_matrices: int,
                       max_flow_amount: int,
                       intensity: float,
                       bw_variation_sqrt: float,
                       flowsperiod: int = 30000,
                       seed: int = None,
                       n_interpolate: int = 1,
                       individual_matrices: bool = False):
    # get a topology
    rng = random.Random()
    with open(topology_file_path, mode='rb') as f:
        topology = networkx.readwrite.read_gml(f)

    min_bw_coef = intensity / bw_variation_sqrt
    max_bw_coef = intensity * bw_variation_sqrt
    generator = DemandMatrixGenerator(min_bw_coef, max_bw_coef, topology, mode=mode, seed=seed)

    # generate some matrices
    # matrices = generator.generate(50, ['1', '2', '3', '4'], ['13', '14', '15', '16'])
    matrices, avg_load, max_load = generator.generate(n_matrices, n_interpolate=n_interpolate)

    if not individual_matrices:
        matrices_lists = [matrices]
    else:
        matrices_lists = [[x] for x in matrices]

    for matrices_idx, matrices in enumerate(matrices_lists):
        if not individual_matrices:
            i_str = ''
        else:
            i_str = f"-{matrices_idx:04}"

        # generate flows using uniform flow generator
        if max_flow_amount > 1:
            flow_generator = UniformFlowGenerator(max_flow_amount, 25000, 700000)
            result = flow_generator.generate(matrices, flowsperiod)
            result_with_param = {
                "params" : {
                    "average_load" : avg_load,
                    "max_load" : max_load,
                    "seed" : seed,
                    "min_bw_coef" : min_bw_coef,
                    "max_bw_coef" : max_bw_coef,
                    "mode" : mode
                },
                "flows" : json.loads(result.json())
            }
            for x in result_with_param['flows']:
                rng.seed(x['flow_id'])
                x['hash'] = rng.randint(0, 2**64 - 1)
            pretty_res = json.dumps(result_with_param, indent=4)
            with open(os.path.splitext(out_file_path)[0] + i_str + '_full' + os.path.splitext(out_file_path)[1], 'w') as f:
                f.write(pretty_res)

        flow_generator_1 = UniformFlowGenerator(1, 25000, 700000)

        # run the generator
        result_1 = flow_generator_1.generate(matrices, flowsperiod)
        result_1_with_param = {
            "params" : {
                "average_load" : avg_load,
                "max_load" : max_load,
                "seed" : seed,
                "min_bw_coef" : min_bw_coef,
                "max_bw_coef" : max_bw_coef,
                "mode" : mode
            },
            "flows" : json.loads(result_1.model_dump_json())
        }



        for x in result_1_with_param['flows']:
            rng.seed(x['flow_id'])
            x['hash'] = rng.randint(0, 2**64 - 1)

        # convert results to a string with optional pretty print - can be removed, only result.json() is needed
        # But results will be unreadable for a human is pretty print is removed

        pretty_res_1 = json.dumps(result_1_with_param, indent=4)

        # open result file and write the flow data

        path_1 = os.path.splitext(out_file_path)[0] + i_str + os.path.splitext(out_file_path)[1]
        with open(path_1, 'w') as f:
            f.write(pretty_res_1)

    # return result

if __name__ == '__main__':
    args = parse_args()

    # set topology_path as path to topology.gml file
    topology_path = args.topology_path
    if topology_path.is_dir():
        topology_path = topology_path / "topology.gml"

    # name of directory containing topology.gml
    topology_name = topology_path.resolve().parent.name
    dat_str = datetime.now().strftime('%Y-%m-%d_%H-%M-%S.%f')[:-3]
    out_file_path_template = f"flows_{topology_name}_{dat_str}.json" if args.out_path is None else args.out_path

    mode = args.mode
    seed = args.seed
    n_matrices = args.matr
    max_flow_amount = args.flows
    intensity = args.intensity
    bw_variation_sqrt = args.bwvar ** 0.5
    flowsperiod = 30000
    n_interpolate = args.interp
    n_matrices_total = (n_matrices - 1) * n_interpolate + 1

    print(f"Topology path: {topology_path}")

    #intens = [0.4]
    #intens = [0.35, 0.4, 0.55, 0.65] # abilene: 35-60%
    #intens = [0.32, 0.4, 0.5, 0.57] # symm16: 40-70%
    #intens = [0.2, 0.3, 0.37] # geant2009/uninett2010: 30-65% ?
    #intens = [0.23, 0.35, 0.45] # bics: 30-65
    #intens = [0.3, 0.4, 0.5, 0.56, 0.62] # arnes: 35-65
    #intens = [0.23, 0.3, 0.35, 0.4, 0.45] # surfnet
    #intens = [0.14, 0.17, 0.24, 0.36] # cogentco: 35, 41, 50, 60

    # intens = [0.2, 0.25, 0.3, 0.34, 0.37] # geant2009/uninett2010: 30-65% ?

    # MLU
    # intens = [0.125, 0.15, 0.175, 0.2] # geant: 0.7-1.5
    # intens = [0.175, 0.23, 0.29, 0.35] # bics: 0.7-1.5
    # intens = [0.13, 0.17, 0.22, 0.27] # uninett2011: 0.7-1.5
    #intens = [0.13, 0.19, 0.25, 0.3] # surfnet
    #intens = [0.13, 0.18, 0.23, 0.28]

    # DAG, MLU, split flows
    #intens = [0.12, 0.15, 0.18, 0.21, 0.24] # geant2009,c,s: 0.8-1.5
    #intens = [0.115, 0.15, 0.185, 0.22] # geant2009,c,s: 0.8-1.5
    #intens = [0.09, 0.113, 0.146, 0.17] # uninett2010 no chains,s: 0.8-1.6
    # intens = [0.15, 0.2, 0.25, 0.3] # surfnet no chains
    # intens = [0.2, 0.266, 0.333, 0.4] # arnes
    #intens = [0.16, 0.21, 0.26, 0.31] # geant2009,nc : 0.8
    # intens = [1.5]
    intens = [args.intensity]

    n = args.nfiles
    individual_matrices = args.splitseq

    if args.ecmp or args.solver:
        topology_path_parent = topology_path.resolve().parent
        experiment_folder = os.path.normpath(topology_path_parent)
        experiment_path_rel = os.path.split(experiment_folder)[-1]
        if os.path.isfile(os.path.join(experiment_folder, "config.yaml")):
            Config.load_config(experiment_folder)
        else:
            Config.load_config(os.path.join(experiment_folder, '../..'))
        config = Config.config()
    if args.ecmp:
        print(f"Calculating ECMP distribution:")
        problem_mode = 1
        if config.split_flows == True:
            print('Splitting demands')
        ecmp_prepared = False

    if args.solver:
        print(f"Calculating LPSolver solution:")
        if config.split_flows == True:
            print('Splitting demands')
        solver_prepared = False

    seed_gen = random.Random(seed)

    for i in range(n):
        cur_seed = seed if i == 0 else seed_gen.randint(0, 65536)
        intensity = intens[i % len(intens)]
        out_file_path_tmpl_spl = os.path.splitext(out_file_path_template)
        i_str = f"{i:04}" if n > 1 else ""
        out_file_path = f"{out_file_path_tmpl_spl[0]}{i_str}{out_file_path_tmpl_spl[1]}"

        generate_synthetic(topology_path, out_file_path, mode, n_matrices,
            max_flow_amount, intensity, bw_variation_sqrt, flowsperiod=flowsperiod,
            seed=cur_seed, n_interpolate=n_interpolate, individual_matrices=individual_matrices)
        print(f"Flows generated succesfully in: {out_file_path}")
        if args.ecmp:
            if not ecmp_prepared:
                out_file_path_rel = os.path.relpath(out_file_path, str(topology_path_parent)) # load any flows just to read topology for the first time
                flowsname_noext = os.path.splitext(out_file_path_rel)[0]
                # log_filename = os.path.join(experiment_folder, f"{flowsname_noext}-{problem_mode}_{dat_str}.log")
                input_data = InputData(experiment_folder, flows_name=out_file_path_rel, ignore_topology_changes=not args.changes)
                topo_obj = input_data.topology
                topo_init = topo_obj.get(0)[0]

                path_calculator = DAGCalculator()
                path_calculator.prepare_iteration(topo_init)
                ecmp_prepared = True

            out_file_path_rel = os.path.relpath(out_file_path, str(topology_path_parent))
            flowsname_noext = os.path.splitext(out_file_path_rel)[0]
            input_data = InputData(experiment_folder, flows_name=out_file_path_rel, ignore_topology_changes=not args.changes)
            topo_obj = input_data.topology
            topo_init = topo_obj.get(0)[0]
            phi_func = PhiCalculator.calculate_phi

            last_changed = None
            topo_changes_times = topo_obj._changed_at
            flows_end = (n_matrices_total - 1) * flowsperiod
            topo_changes_end = topo_changes_times[-1] if len(topo_changes_times) > 0 else 0
            t_end = max(flows_end, topo_changes_end)
            t = 0
            iteration = 0
            while t <= t_end:
                topo, last_changed_new = topo_obj.get(t)
                if last_changed_new != last_changed:
                    path_calculator = DAGCalculator()
                    path_calculator.prepare_iteration(topo)
                last_changed = last_changed_new
                flows = input_data.flows.get(t)
                print(f"number of flows: {len(flows)}")
                filename_prefix = os.path.join(experiment_folder, f"{flowsname_noext}-ecmp-{problem_mode}-{t}_{dat_str}")

                alg = PhiTest(problem_mode, topo, flows, path_calculator, phi_func, experiment_path_rel, t)
                weights, phi, cbw, Bandwidth, util, util_mean, util_max = alg.run()

                np.save(f'{filename_prefix}_population.npy', np.array([weights]))
                np.savez(f'{filename_prefix}_solution.npz', solution=weights, mlu=np.array([util_max]), phi=np.array([phi]))
                print(f"Iteration: {iteration}, t={t:8}, ECMP: AVG BANDWIDTH TAKEN: {util_mean}, MAX BANDWIDTH TAKEN: {util_max}, phi = {phi}")
                print()
                topo_next_changes = [tt for tt in topo_changes_times if tt > t]
                topo_next_change = topo_next_changes[0] if len(topo_next_changes) > 0 else t + flowsperiod + 1
                t = min(t + flowsperiod, topo_next_change)
                iteration += 1

        if args.solver:
            if not solver_prepared:
                out_file_path_rel = os.path.relpath(out_file_path, str(topology_path_parent)) # load any flows just to read topology for the first time
                flowsname_noext = os.path.splitext(out_file_path_rel)[0]
                # log_filename = os.path.join(experiment_folder, f"{flowsname_noext}-{problem_mode}_{dat_str}.log")
                input_data = InputData(experiment_folder, flows_name=out_file_path_rel, ignore_topology_changes=not args.changes)
                topo_obj = input_data.topology
                topo_init = topo_obj.get(0)[0]
                topo_init_conv, node_map = convert_topo(topo_init)
                path_calculator2 = DAGCalculator()
                path_calculator2.prepare_iteration(topo_init_conv)
                solver = LPSolver(topo_init_conv, node_map, path_calculator2)
                solver_prepared = True

            out_file_path_rel = os.path.relpath(out_file_path, str(topology_path_parent))
            flowsname_noext = os.path.splitext(out_file_path_rel)[0]
            flows_demand = load_flows(os.path.join(experiment_folder, out_file_path_rel))
            if args.flows > 1: flows2 = load_flows(os.path.join(experiment_folder, os.path.splitext(out_file_path_rel)[0] + '_full' + '.json'))

            last_changed = None
            topo_changes_times = topo_obj._changed_at
            flows_end = (n_matrices_total - 1) * flowsperiod
            topo_changes_end = topo_changes_times[-1] if len(topo_changes_times) > 0 else 0
            t_end = max(flows_end, topo_changes_end)
            t = 0
            iteration = 0
            while t <= t_end:
                topo_, last_changed_new = topo_obj.get(t)
                topo, node_map = convert_topo(topo_)
                if last_changed_new != last_changed:
                    path_calculator2 = DAGCalculator()
                    path_calculator2.prepare_iteration(topo)
                    solver = LPSolver(topo, node_map, path_calculator2)
                last_changed = last_changed_new
                flows2_t = get_flows_at(flows_demand, t, node_map)
                print(f"number of flows: {len(flows2_t)}")
                filename_prefix2 = os.path.join(experiment_folder, f"{flowsname_noext}-opt-{t}_{dat_str}")
                opt_value, solution = solver.solve(flows2_t, save=True)

                hash_phi = ''
                if args.flows > 1:
                    flows_t = get_flows_at(flows2, t, node_map, numbers=False)

                    hash_phi = test_flow_hash(solution, topo_, flows_t)

                np.save(f'{filename_prefix2}.npy', np.array([opt_value]))
                print(f"Iteration: {iteration}, t={t:8}, OPTIMUM: MAX BANDWIDTH TAKEN: {opt_value}, {hash_phi}")
                print()
                topo_next_changes = [tt for tt in topo_changes_times if tt > t]
                topo_next_change = topo_next_changes[0] if len(topo_next_changes) > 0 else t + flowsperiod + 1
                t = min(t + flowsperiod, topo_next_change)
                iteration += 1
