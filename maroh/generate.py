from generator.generator import UniformFlowGenerator
from generator.demand_generator import DemandMatrixGenerator
from generator.parsers import sndlib_brain as sndlib_brain_parser
from generator.weightmatrix_test import PhiTest
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
    parser.add_argument('--intensity', type=float, default=0.3,
                        help='Increasing/decreasing this value increases/decreases mean occupied bandwith.'
                        'Choose manually until achieving desired value of mean occupied bandwith (see "average bandwidth taken" in the end of console output).')
                        # TODO: make the selection automatic
    parser.add_argument('--bwvar', type=float, default=12.25,
                        help='Bandwidth variation square root, sets sqrt(max_bw_coef/min_bw_coef) for DemandMatrixGenerator arguments max_bw_coef, min_bw_coef')
    parser.add_argument('--matr', type=int, default=1,
                       help='number of matrices')

    parser.add_argument('--out_path', type=Path, default=None,
                       help='json output file name')

    parser.add_argument('--ecmp', action='store_true', help='Calculate ECMP distribution after generation')

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
                       individual_matrices: bool = False):
    # get a topology
    rng = random.Random()
    with open(topology_file_path, mode='rb') as f:
        topology = networkx.readwrite.read_gml(f)

    coef = 1.3 # may need to change manually for bw_variation_sqrt values other than 3.5
    mean_bw_expected = intensity / (bw_variation_sqrt * coef)
    min_bw_coef = mean_bw_expected / bw_variation_sqrt
    max_bw_coef = mean_bw_expected * bw_variation_sqrt
    generator = DemandMatrixGenerator(min_bw_coef, max_bw_coef, topology, mode=mode, seed=seed)

    # generate some matrices
    # matrices = generator.generate(50, ['1', '2', '3', '4'], ['13', '14', '15', '16'])
    matrices, avg_load, max_load = generator.generate(n_matrices)

    # generate flows using uniform flow generator
    flow_generator = UniformFlowGenerator(max_flow_amount, 25000, 700000)

    if not individual_matrices:
        matrices_lists = [matrices]
    else:
        matrices_lists = [[x] for x in matrices]

    for matrices_idx, matrices in enumerate(matrices_lists):
        # run the generator
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

        # convert results to a string with optional pretty print - can be removed, only result.json() is needed
        # But results will be unreadable for a human is pretty print is removed
        pretty_res = json.dumps(result_with_param, indent=4)

        if not individual_matrices:
            out_file_path_mod = out_file_path
        else:
            out_file_path_name, out_file_path_ext = os.path.splitext(out_file_path)
            out_file_path_mod = f"{out_file_path_name}-{matrices_idx:04}{out_file_path_ext}"
        else:


        # open result file and write the flow data
        with open(out_file_path_mod, 'w') as f:
            f.write(pretty_res)

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
    out_file_path = f"flows_{topology_name}_{dat_str}.json" if args.out_path is None else args.out_path

    mode = args.mode
    seed = args.seed
    n_matrices = args.matr
    max_flow_amount = args.flows
    intensity = args.intensity
    bw_variation_sqrt = args.bwvar ** 0.5
    flowsperiod = 30000


    print(f"Topology path: {topology_path}")
    generate_synthetic(topology_path, out_file_path, mode, n_matrices,
        max_flow_amount, intensity, bw_variation_sqrt, flowsperiod=flowsperiod, seed=seed)
    print(f"Flows generated succesfully in: {out_file_path}")
    if args.ecmp:
        print(f"Calculating ECMP distribution:")
        topology_path_parent = topology_path.resolve().parent
        out_file_path_rel = os.path.relpath(out_file_path, str(topology_path_parent))

        experiment_folder = os.path.normpath(topology_path_parent)
        experiment_path_rel = os.path.split(experiment_folder)[-1]
        problem_mode = 1

        dat_str = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

        Config.load_config(os.path.join(experiment_folder, '../..'))
        config = Config.config()
        flowsname_noext = os.path.splitext(out_file_path_rel)[0]
        log_filename = os.path.join(experiment_folder, f"{flowsname_noext}-{problem_mode}_{dat_str}.log")
        input_data = InputData(experiment_folder, flows_name=out_file_path_rel, ignore_topology_changes=True)
        topo = input_data.topology.get(0)[0]

        phi_func = PhiCalculator.calculate_phi
        path_calculator = DAGCalculator()

        path_calculator.prepare_iteration(topo)

        for tim in range(0, n_matrices * flowsperiod, flowsperiod):
            print(f"env time: {tim}")
            flows = input_data.flows.get(tim)
            print(f"number of flows: {len(flows)}")
            filename_prefix = os.path.join(experiment_folder, f"{flowsname_noext}-ecmp-{problem_mode}-{tim}_{dat_str}")

            alg = PhiTest(problem_mode, topo, flows, path_calculator, phi_func, experiment_path_rel, tim)
            weights, phi, cbw, Bandwidth, util, util_mean, util_max = alg.run()

            np.save(f'{filename_prefix}_population.npy', np.array([weights]))
            np.savez(f'{filename_prefix}_solution.npz', solution=weights, phi=np.array([phi]))

            # print(f"Weights matrix:\n {weights}")
            print(f"Phi function value: {phi:7.5f}")
            # print(f"Links load:\n {util}")
            print(f"Avg link load: {util_mean}")
            print(f"Max link load: {util_max}")
