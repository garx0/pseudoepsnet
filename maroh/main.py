import importlib
import sys
import os
import shutil
import argparse
import json
import torch
import random
import numpy as np
import torch
from datetime import datetime
from dte_stand.config import Config
from dte_stand.history import HistoryTracker
from dte_stand.logger import init_logger
from dte_stand.phi_calculator import PhiCalculator
from dte_stand.controller import ExperimentController, TrainingController, RandomExperimentController, CalcActionsController

def dynamic_import_function(object_path):
    module, object, function_name = object_path.rsplit('.', 2)
    imported_module = importlib.import_module(module)
    obj_class = getattr(imported_module, object)
    return getattr(obj_class, function_name)


def dynamic_import(object_path: str, **module_kwargs):
    module, object = object_path.rsplit('.', 1)
    imported_module = importlib.import_module(module)
    obj_class = getattr(imported_module, object)
    return obj_class(**module_kwargs)


def remove_link_experiment():
    all_avg = []
    for _ in range(300):
        phi_func = dynamic_import_function(config.phi)
        path_calculator = dynamic_import(config.path_calculator)
        hash_function = dynamic_import(config.hash_function, path_calculator=path_calculator,
                                       debug_check_cycles=config.debug_check_cycles)
        algo = dynamic_import(config.algorithm, hash_function=hash_function, phi_func=phi_func,
                              experiment_dir=result_path,
                              model_dir=args.model)

        controller = ExperimentController(
                args.experiment_folder, config.lsdb_period, config.iterations, hash_function,
                algo, path_calculator, phi_func, result_path)
        _, start1, end1, start2, end2 = controller.run()
        all_avg.append((PhiCalculator.get_average(), start1, end1, start2, end2))

    avg_json = json.dumps(all_avg)
    with open(os.path.join(result_path, 'avg.json'), 'w') as f:
        f.write(avg_json)


def normal_experiment():
    controller.run()
    PhiCalculator.plot_full(all_iterations=True)


def ecmp_experiment():
    PhiCalculator.init_tracker()
    controller.run()
    PhiCalculator.end_episode()


def set_seed(seed = 0):
    #tf.keras.utils.set_random_seed(seed)
    #tf.random.set_seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def set_determ():
    #tf.config.experimental.enable_op_determinism()
    torch.backends.deterministic = True
    torch.backends.benchmark = False
    torch.use_deterministic_algorithms(True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('experiment_folder', type=str,
                        help='path to folder that contains experiment input data\n'
                             'data_examples in this repository is an example of the structure of this folder')
    parser.add_argument('-n', '--name', type=str, default='',
                        help='description of the experiment')
    parser.add_argument('-m', '--model', type=str, default=None,
                        help='path to folder of saved model to load into current experiment')
    parser.add_argument('--multi', action='store_true', help='Apply multiple actions (SA MAROH)')
    parser.add_argument('-d', '--deterministic', action='store_true', help='Run experiment in deterministic mode.')
    parser.add_argument('-s', '--seed', type=int, help='Set global random seeds of numpy/python to given value')
    parser.add_argument('-pt', '--pytorch', action='store_true', help='Run experiment on PyTorch')
    parser.add_argument('-tm', '--time-mem', action='store_true', help='Write memory usage and episode time, to parse use mem_graph.py, old memory_usage.txt will be deleted')
    parser.add_argument('-t', '--train_mode', action='store_true', help='Train mode')
    parser.add_argument('-c', '--cfg', type=str, default='',
                        help='config')
    parser.add_argument('--memory', type=str, default=None,
                        help='path to .npz file with memory')
    parser.add_argument('--states', type=str, default=None, help='Directory with sequence of states for calculating actions')
    args = parser.parse_args()

    if args.deterministic:
        set_determ()
        set_seed(0)

    if args.seed:
        set_seed(args.seed)

    if os.path.exists("memory_usage.txt"):
        os.remove("memory_usage.txt")

    Config.load_config(args.experiment_folder, modifier=args.cfg)
    config = Config.config()

    date_str = datetime.now().strftime('%Y-%m-%d,%H-%M-%S.%f')[:-3]
    sa_str = "sa" if args.multi else "non-sa"
    pt_str = "_pt" if args.pytorch else ""
    train_mode = args.train_mode
    states_path = args.states
    result_path = os.path.join(args.experiment_folder, f"exp_{date_str}_{args.name}_{sa_str}{pt_str}")
    os.mkdir(result_path)
    shutil.copy(os.path.join(args.experiment_folder, f'config{args.cfg}.yaml'), result_path)
    PhiCalculator.set_plot_folder(result_path)
    HistoryTracker.set_result_folder(result_path)

    init_logger(os.path.join(result_path, config.log_path), config.log_level, ['matplotlib'])

    phi_func = dynamic_import_function(config.phi)
    path_calculator = dynamic_import(config.path_calculator)
    hash_function = dynamic_import(config.hash_function, path_calculator=path_calculator,
                                   debug_check_cycles=config.debug_check_cycles)
    algo = dynamic_import(config.algorithm, path_calculator=path_calculator, hash_function=hash_function, phi_func=phi_func, experiment_dir=result_path, model_dir=args.model, multi_actions=args.multi, pt_flag=args.pytorch, check_mem_time=args.time_mem)

    if train_mode:
        controller = TrainingController(args.experiment_folder, hash_function, algo, path_calculator, phi_func, result_path, args.memory)
    elif states_path is not None:
        controller = CalcActionsController(args.experiment_folder, hash_function, algo, path_calculator, phi_func, result_path, states_path=states_path)
    else:
        shutil.copy(os.path.join(args.experiment_folder, 'flows.json'), result_path)
        controller = ExperimentController(
            args.experiment_folder, config.lsdb_period, config.iterations, hash_function,
            algo, path_calculator, phi_func, result_path, args.memory)

    normal_experiment()
    # ecmp_experiment()
    # remove_link_experiment()
