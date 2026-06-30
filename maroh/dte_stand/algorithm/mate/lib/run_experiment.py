import logging
import os
import time

import networkx
import logging
from networkx import diameter as graph_diameter

from dte_stand.algorithm.mate.agents.sa_ppo_agent_pt import SaPPOAgent as PTSaPPOAgent
from dte_stand.algorithm.mate.environment.environment import Environment
from dte_stand.config import Config
from dte_stand.algorithm.mate.config import MateConfig

LOG = logging.getLogger(__name__)


# @gin.configurable
class Runner(object):
    # Runner initializes and stores environment and agent,
    # and can initiate agent training or steps (weights calculations)
    # Runner also takes path calculator, hash function, and phi as input, as these are used by the environment
    # This Runner is MateAlgorithm-specific !
    def __init__(self,
                 path_calculator,
                 hash_function,
                 phi_func,
                 algorithm='PPO',
                 reload_model=False, # load model
                 model_dir='checkpoints/training/gravity_1/PPO_agg_period100/clip0.2/gamma0.95/episode',
                 only_eval=False, # experiment mode (no critic)
                 base_dir='dte_stand/algorithm/mate/logs',
                 checkpoint_dir='dte_stand/algorithm/mate/checkpoints',
                 save_checkpoints=True,
                 multi_actions=False,
                 pt_flag=False,
                 check_mem_and_time=False,
                 memory_path=None,
                 states_path=None):

        self.check_mem_and_time = check_mem_and_time
        config = Config.config()
        mate_config = MateConfig.parse_obj(config.alg_cfg)
        # try:
        #     message_iterations = (
        #             mate_config.message_iterations if mate_config.message_iterations > 0
        #             else graph_diameter(topology_object)
        #     )
        # except networkx.NetworkXError:
        #     LOG.exception('Message iterations is not given in config '
        #                   'and topology does not have all links in both directions')
        #     raise Exception(
        #             'Message iterations is not given in config '
        #             'and topology does not have all links in both directions.\n'
        #             'Either set message_iterations to a positive value in mate section in config '
        #             'or the topology must be strongly connected in both directions.')

        message_iterations = mate_config.message_iterations

        mate_actions = mate_config.actions
        self.save_checkpoints = save_checkpoints
        self.hash_function = hash_function
        self.env = Environment(
                None, # no initial topology
                path_calculator,
                self.hash_function,
                mate_actions,
                phi_func,
                base_reward=mate_config.reward,
                reward_computation=mate_config.reward_computation,
                min_weight=mate_config.min_weight,
                max_weight=mate_config.max_weight,
                weights_initialization=mate_config.init_weights,
                clip_state=mate_config.clip_state
        )

        # if multi_actions:
        #     agent = PTSaPPOAgent
        # else:
        #     agent = PTPPOAgent
        agent = PTSaPPOAgent

        if algorithm == 'PPO':
            self.agent = agent(
                    self.env,
                    mate_config.actor_cfg,
                    mate_actions,
                    phi_func,
                    checkpoint_dir=checkpoint_dir,
                    message_iterations=message_iterations,
                    plot_period=config.plot_period,
                    horizon=mate_config.horizons,
                    batch_size=mate_config.batch_size,
                    eval_period=mate_config.eval_period,
                    gamma=mate_config.gamma,
                    gae_lambda=mate_config.gae_lambda,
                    lr_actor=mate_config.lr_actor,
                    lr_critic=mate_config.lr_critic,
                    greedy_eplison=mate_config.greedy_epsilon,
                    save_checkpoints=save_checkpoints,
                    check_mem_and_time=self.check_mem_and_time,
                    episodes=mate_config.episodes,
                    change_sample_period=mate_config.change_sample_period,
                    n_without_update=mate_config.n_without_update,
                    memory_path=memory_path,
                    states_path=states_path,
                    random_sample=mate_config.random_sample,
                    entropy_loss_factor=mate_config.entropy_coef,
                    max_grad_norm=mate_config.grad_clip
            )
        else:
            assert False, 'RL Algorithm %s is not implemented' % algorithm
        self.base_dir = base_dir
        self.checkpoint_base_dir = checkpoint_dir
        self.only_eval = only_eval

        if reload_model:
            self.agent.load_saved_model(model_dir, only_eval)

        if self.save_checkpoints and (not os.path.exists(self.checkpoint_base_dir)):
            os.makedirs(self.checkpoint_base_dir)

    def update(self, topology, checkpoint_dir, save_model):
        #
        self.save_checkpoints = save_model
        self.agent.save_checkpoints = save_model
        if self.save_checkpoints:
            self.checkpoint_base_dir = checkpoint_dir
            self.agent.set_checkpoint_dir(checkpoint_dir)
            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)
        self.env.update_topology(topology)
        self.agent.change_sample = False


    def run_step(self, topology, current_flows, horizons = None, hash_weights=None, topo_changed=False):
        # calculate weights for given topo and TM (for a pre-trained agent)
        if self.check_mem_and_time:
            full_time = time.time()
        hash_weights = self.agent.run_step(topology, current_flows, horizons, hash_weights=hash_weights, topo_changed=topo_changed)
        if self.check_mem_and_time:
            full_time = time.time() - full_time
            print("\n----------------------------\n")
            print(f"full weight calculation time: {full_time:.2f} sec")
            print("\n----------------------------\n")

        return hash_weights

    def run_training(self, exp_dir):
        # train the agent on a dataset
        if self.check_mem_and_time:
            full_time = time.time()
        hash_weights = self.agent.train_and_evaluate(exp_dir)
        if self.check_mem_and_time:
            full_time = time.time() - full_time
            print("\n----------------------------\n")
            print(f"full training time: {full_time:.2f} sec")
            print("\n----------------------------\n")

        return hash_weights

    def run_calc_actions(self, exp_dir):
        # train the agent on a dataset
        if self.check_mem_and_time:
            full_time = time.time()
        hash_weights = self.agent.calc_actions(exp_dir)
        if self.check_mem_and_time:
            full_time = time.time() - full_time
            print("\n----------------------------\n")
            print(f"full training time: {full_time:.2f} sec")
            print("\n----------------------------\n")

        return hash_weights
