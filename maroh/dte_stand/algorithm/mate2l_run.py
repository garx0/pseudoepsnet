import networkx
import os
from typing import Optional, Callable
from dte_stand.hash_function.base import BaseHashFunction
from dte_stand.algorithm.base import BaseAlgorithm
from dte_stand.data_structures import HashWeights, Flow
from dte_stand.algorithm.mate2l.lib.run_experiment import Runner
from dte_stand.algorithm.mate2l.config import MateConfig

import logging
LOG = logging.getLogger(__name__)


class Mate2LAlgorithm(BaseAlgorithm):
    # MARL + GNN algorithm. Needs training
    def __init__(self, path_calculator, hash_function: BaseHashFunction, phi_func: Callable, experiment_dir: str, model_dir: str, multi_actions: bool = False, pt_flag : bool = False, check_mem_time: bool = False):
        super().__init__(hash_function, phi_func, experiment_dir, model_dir)
        self._runner: Optional[Runner] = None
        self._multi_actions = multi_actions
        self.pt_flag = pt_flag
        self.check_mem_time = check_mem_time
        self.path_calculator = path_calculator

    def step(self, horizons, topology: networkx.MultiDiGraph, path_calculator, flows: list[Flow],
             iteration_num: int = 0, train = False, save_model: bool = False, exp_dir = None, hash_weights: HashWeights = None, topo_changed=False, memory_path=None) -> HashWeights:
        if not self._runner:
            self._runner = Runner(
                    path_calculator, self._hash_function, self._phi,
                    checkpoint_dir=os.path.join(self._experiment_dir, 'model'),
                    save_checkpoints=False, reload_model=bool(self._model_dir), model_dir=self._model_dir,
                    multi_actions=self._multi_actions, pt_flag = self.pt_flag, check_mem_and_time=self.check_mem_time, only_eval=True, memory_path=memory_path
                    )
        #self._runner.update(topology, os.path.join(self._experiment_dir, 'model'), save_model)

        return self._runner.run_step(topology, flows, horizons, hash_weights=hash_weights, topo_changed=topo_changed)
        #return self._runner.run_training(exp_dir)

    def train_algorithm(self, exp_dir, memory_path=None):
        if not self._runner:
            self._runner = Runner(
                    self.path_calculator, self._hash_function, self._phi,
                    checkpoint_dir=os.path.join(self._experiment_dir, f'results', 'model'),
                    save_checkpoints=True, reload_model=bool(self._model_dir), model_dir=self._model_dir,
                    multi_actions=self._multi_actions, pt_flag = self.pt_flag, check_mem_and_time=self.check_mem_time,
                    memory_path=memory_path
                    )

        #return self._runner.run_experiment(topology, flows, exp_dir)
        return self._runner.run_training(exp_dir)

    def calc_actions(self, exp_dir, states_path):
        if not self._runner:
            self._runner = Runner(
                    self.path_calculator, self._hash_function, self._phi,
                    checkpoint_dir=os.path.join(self._experiment_dir, f'results', 'model'),
                    save_checkpoints=True, reload_model=bool(self._model_dir), model_dir=self._model_dir,
                    multi_actions=self._multi_actions, pt_flag = self.pt_flag, check_mem_and_time=self.check_mem_time,
                    states_path=states_path
                    )

        #return self._runner.run_experiment(topology, flows, exp_dir)
        return self._runner.run_calc_actions(exp_dir)
