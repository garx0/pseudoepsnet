import copy
import csv
import os
import gc
import random
import typing as t
import numpy as np
import torch
import dill
import networkx as nx
import re
from collections import defaultdict
from time import sleep

from dte_stand.phi_calculator import PhiCalculator
from dte_stand.algorithm.mate2l.environment.environment import Environment
from dte_stand.algorithm.mate2l.lib.actor_pt import Actor
from dte_stand.algorithm.mate2l.lib.critic_pt import Critic
from dte_stand.algorithm.mate2l.config import MateActions, SaActorCfg
from dte_stand.history import HistoryTracker
from dte_stand.algorithm.mate2l.utils.memory_checker import write_memory_usage
from dte_stand.data_structures.flows import Flows

N_IO_TRIES = 12

def check_gradients(model):
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_mean = param.grad.abs().mean().item()
            print(f"{name}: grad = {grad_mean}")
        else:
            print(f"{name}: no grad")

class MarohResultsParser:
    def __init__(self, path, parameter_name):
        self._folder_path = path
        self._results = None
        self._max_episode = 0
        self._parameter_name = parameter_name
        self._parse_folder()

    def _parse_file(self, file_obj, start):
        data_list = np.load(file_obj)
        if self._results is None:
            self._results = defaultdict(lambda: defaultdict(lambda: []))
        for subparam, subparam_data_list in data_list.items():
            episode_index = start
            for episode_data in subparam_data_list:
                self._results[subparam][episode_index] = episode_data
                episode_index += 1
            if episode_index > self._max_episode:
                self._max_episode = episode_index

    def _parse_folder(self, path=None):
        cur_path = path if path else self._folder_path
        files_parsed = 0
        with os.scandir(cur_path) as files:
            for file in files:
                if file.is_dir(follow_symlinks=False):
                    continue
                if not file.is_file():
                    continue
                match = re.search(f'({self._parameter_name})_([0-9]+)-([0-9]+)\.np[y|z]', file.name)
                if not match:
                    continue
                with open(file.path, 'rb') as f:
                    self._parse_file(f, start=int(match[2]))
                    files_parsed += 1

def load_states(path, load_actions=False):
    '''
    result_path: directory containing states_*-*.npz file(s) and action_gr_*-*.npz file(s)
    '''
    values = {}
    params = ["states", "actions_gr"] if load_actions else ["states"]
    for param in params:
        folder = MarohResultsParser(path, parameter_name=param)
        if folder._results is None:
            raise Exception(f"couldn't read states_*-*.npz file(s) and action_gr_*-*.npz file(s) from {path}")
        for subparam, res_subparam in folder._results.items():
            max_episode = len(res_subparam)
            values[subparam] = [res_subparam[episode] for episode in range(0, max_episode)]
    states = {}
    for key in values.keys():
        if not load_actions or key != 'actions_gr':
            states[int(key)] = np.array(values[key])
    if load_actions:
        actions = np.array(values['actions_gr'])
        return states, actions
    else:
        return states

class SaPPOAgent(object):
    """An implementation of a GNN-based PPO Agent"""

    def __init__(self,
                 env: Environment,
                 actor_cfg: t.Optional[dict],
                 action_config: MateActions,
                 phi_func,
                 message_iterations,
                 eval_env_type=['Test'],
                 plot_period=1000,
                 num_eval_samples=1, # was 3 in MATE before
                 clip_param=0.25,
                 critic_loss_factor=0.5,
                 entropy_loss_factor=0.001,
                 normalize_advantages=True,
                 max_grad_norm=1.0,
                 gamma=0.99,
                 gae_lambda=0.95,
                 lr_actor=0.0003,
                 lr_critic=0.0003,
                 horizon=100,
                 batch_size=64, # was 25 in MATE before
                 epochs=3,
                 last_training_sample=1,
                 eval_period=50,
                 max_evals=5,
                 select_max_action=False,
                 optimizer=torch.optim.Adam,
                 change_traffic=True,
                 change_sample_period=15,
                 base_dir='logs',
                 checkpoint_dir='checkpoints',
                 save_checkpoints=True,
                 greedy_eplison=1,
                 check_mem_and_time=False,
                 episodes=5000,
                 n_without_update=1,
                 memory_path=None,
                 states_path=None,
                 random_sample=True):

        self.phi = phi_func
        self.env = env
        self.actor_cfg = SaActorCfg(**actor_cfg) if actor_cfg is not None else SaActorCfg()
        self.action_config = action_config
        self.eval_env_type = eval_env_type
        self.num_eval_samples = num_eval_samples
        self.clip_param = clip_param
        self.check_mem_and_time = check_mem_and_time

        self.current_gml = None
        self.topo_dict = dict()
        self.new_topo = True
        self.samples = list()

        self.current_topology = env.G
        self.current_flows = None

        # Strategy of agents
        self.strategies = ["NORMAL", "EQUAL", "RANDOM"]
        self.strategy = "NORMAL"

        self.train_with_memory = False #True = use with clone, False = no memory while training
        self.actor = None

        self.dyn_msg_iterations = message_iterations < 0
        self.message_iterations = message_iterations

        self.episodes = episodes
        self.n_without_update = n_without_update

        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            print("CUDA (GPU) is available")
        else:
            self.device = torch.device("cpu")
            print("CUDA (GPU) is not available, using CPU")
        self._get_actor_critic_functions(memory_path=memory_path)
        self._can_compile = self.actor.can_compile

        self.optimizer = optimizer([
                                   { "params": self.actor.parameters(), "lr": lr_actor, "betas": (0.9,0.999), "eps":0.00001 },
                                   { "params": self.critic.parameters(), "lr": lr_critic, "betas": (0.9,0.999), "eps":0.00001 }
                                   ])
        self.critic_loss_factor = critic_loss_factor
        self.entropy_loss_factor = entropy_loss_factor
        self.normalize_advantages = normalize_advantages
        self.max_grad_norm = max_grad_norm

        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.given_horizon = horizon
        self.horizon = horizon
        self.epochs = epochs
        self.batch_size = batch_size
        self.last_training_sample = last_training_sample
        self.eval_period = eval_period
        self.max_evals = max_evals
        self.select_max_action = select_max_action
        self.change_traffic = change_traffic
        self.change_sample_period = change_sample_period
        self.eval_step = 0
        self.eval_episode = 0
        self.episode = 0
        self.base_dir = base_dir
        self.checkpoint_dir = checkpoint_dir
        self.save_checkpoints = save_checkpoints if self.n_without_update != 0 else False
        self.reload_model = False
        self.change_sample = False
        self.eps = greedy_eplison
        self.plot_period = plot_period
        self.tracker = HistoryTracker()
        self.set_experiment_identifier(False)
        self.states_logging = self.actor.states_logging
        if self.states_logging:
            self.log_actions_gr = [] # greedy actions
            if self.actor.memory_used:
                self.log_actions_gr_true = [] # greedy actions calculated by readout

        self.states_path = states_path
        self.random_sample = random_sample


    def load_dataset(self, path):
        dataset = dict()
        for x in next(os.walk(path))[1]:
            h = os.path.join(path, x)
            dataset[h] = []
            for y in next(os.walk(h))[2]:
                if y.endswith('.json'): dataset[h].append(y)
        return dataset

    def set_dataset(self, path):
        self.train_path = os.path.join(path, 'train')
        self.train_set = self.load_dataset(self.train_path)
        self.test_path = os.path.join(path, 'test')
        self.test_set = self.load_dataset(self.test_path)

    def load_sample(self, topo_dir, tm_dir, tm):
        gml = os.path.join(topo_dir, 'topology.gml')
        self.new_topo = self.current_gml != gml
        self.current_gml = gml
        # topo = nx.MultiDiGraph(nx.read_gml(gml))
        topo = None
        for _ in range(N_IO_TRIES):
            try:
                topo = nx.MultiDiGraph(nx.read_gml(gml))
                break
            except Exception as e:
                print(f"IOERROR: {e}")
                sleep(5)
        if topo is None:
            raise Exception("couldn't read {gml} due to repeated IOERROR")
        flows = Flows(os.path.join(tm_dir, tm)).get(0)
        return topo, flows

    def update_sample(self, training_episode=None):
        topo_dir = list(self.train_set.items())[torch.randint(0, len(self.train_set), (1,))][0]
        if self.random_sample or training_episode is None:
            tm = self.train_set[topo_dir][torch.randint(0, len(self.train_set[topo_dir]), (1,))]
        else:
            tm = self.train_set[topo_dir][training_episode % len(self.train_set[topo_dir])]
        self.current_topology, self.current_flows = self.load_sample(topo_dir, topo_dir, tm)
        if self.dyn_msg_iterations:
            self.message_iterations = nx.diameter(self.current_topology)
            self.actor.update_message_iterations(self.message_iterations)
            self.critic.update_message_iterations(self.message_iterations)
            print('message_iterations =', self.message_iterations)
        print('sample changed to', topo_dir, tm)



    def set_checkpoint_dir(self, checkpoint_dir):
        self.checkpoint_dir = checkpoint_dir

    def _get_actor_critic_functions(self, memory_path=None):
        self.actions = {}
        self.num_actions = 0

        if self.action_config.addition.action:
            self.actions[self.num_actions] = '+'
            self.num_actions += 1
        if self.action_config.subtraction.action:
            self.actions[self.num_actions] = '-'
            self.num_actions += 1
        if self.action_config.multiplication.action:
            self.actions[self.num_actions] = '*'
            self.num_actions += 1
        if self.action_config.multiplication2.action:
            self.actions[self.num_actions] = '*2'
            self.num_actions += 1
        if self.action_config.division.action:
            self.actions[self.num_actions] = '/'
            self.num_actions += 1
        if self.action_config.division2.action:
            self.actions[self.num_actions] = '/2'
            self.num_actions += 1
        if self.action_config.zero.action:
            self.actions[self.num_actions] = '0'
            self.num_actions += 1


        memory = np.load(memory_path) if memory_path is not None else None
        self.actor = Actor(self.actor_cfg, self.env.G, adj_matrix=self.env.get_adj_matrix(), num_actions=self.num_actions, num_features=self.env.num_features,
                           message_iterations=self.message_iterations, memory_state=memory, device=self.device).to(self.device)
        # self.actor = torch.compile(self.actor)
        self.critic = Critic(self.env.G, num_features=self.env.num_features, message_iterations=self.message_iterations, device=self.device).to(self.device)
        # self.critic = torch.compile(self.critic)
        # on rhombus with CPU with compile becomes slower



    def gae_estimation(self, rewards, values, last_value):
        last_gae_lambda = 0
        advantages = np.zeros_like(values, dtype=np.float32)
        for i in reversed(range(self.horizon)):
            if i == self.horizon - 1:
                next_value = last_value
            else:
                next_value = values[i + 1]
            delta = rewards[i] + self.gamma * next_value - values[i]
            advantages[i] = last_gae_lambda = delta + self.gamma * self.gae_lambda * last_gae_lambda
        returns = values + advantages
        if self.normalize_advantages:
            advantages = (advantages - np.mean(advantages)) / (np.std(advantages) + 1e-8)
        return returns, advantages


    def run_episode(self, new_sample = False, horizons = None, hash_weights = None, keep_memory = False, select_max=False): # TODO: no critic during eval
        # run one episode
        if horizons == None: horizons = self.horizon
        if new_sample:
            self.env.path_calculator.gml = self.current_gml
            self.env.reset(self.current_topology, self.current_flows, hash_weights=hash_weights, new_topo=self.new_topo, equal=select_max)
            if keep_memory == False or self.new_topo:
                self.actor.update_graph(self.env.G, keep_memory=keep_memory)
                self.critic.update_graph(self.env.G)
        else:
            self.env.reset()
        state = self.env.get_state()
        states = np.zeros((horizons, self.env.n_links *
                           self.actor.num_features), dtype=np.float32)
        actions = np.zeros([horizons, self.env.G.number_of_edges()], dtype=np.float32)
        rewards = np.zeros(horizons, dtype=np.float32)
        log_probs = np.zeros([horizons, self.env.G.number_of_edges()], dtype=np.float32)
        values = np.zeros(horizons, dtype=np.float32)
        weights = []
        probabilities = []

        self.episode_min_phi = 10.0

        print('Initial phi:', self.phi(self.env.G, eval=True))

        for t in range(horizons):
            if self.strategy == "EQUAL":
                action_types = [self.actions[0]] * (self.env.G.number_of_edges())
                next_state, reward = self.env.multiple_step(action_types, last=(True if t == self.horizon - 1 else False))
            elif self.strategy == "RANDOM":
                action_types = [self.actions[random.randint(0, len(self.actions) - 1)]] * (self.env.G.number_of_edges())
                next_state, reward = self.env.multiple_step(action_types, last=(True if t == self.horizon - 1 else False))
            else:
                action, log_prob, raw_probs, action_greedy, action_greedy_true = self.act(state, select_max=select_max)
                if self.states_logging and not self.actor.eval_mode:
                    self.log_actions_gr.append(action_greedy.detach().cpu().numpy())
                    if self.actor.memory_used and not self.actor.training:
                        self.log_actions_gr_true.append(action_greedy_true.detach().cpu().numpy())
                value = self.run_critic(state)
                action_numpy = action.detach().cpu().numpy()
                # print(actions_numpy)
                action_types = [self.actions[a] for a in action_numpy]
                # act_val = action_numpy // self.num_actions
                next_state, reward = self.env.multiple_step(action_types, last=(True if t == horizons - 1 else False))

                if select_max == True:
                    phi_val = self.phi(self.env.G, eval=True)
                    print(t, "act:", action_numpy, "rwd:", reward, "phi:", phi_val)
                    if phi_val < self.episode_min_phi:
                        self.episode_min_phi = phi_val
                # probabilities.append(raw_probs)
                # weights.append(copy.deepcopy(self.env.weights))
                states[t] = state
                actions[t] = action.detach().cpu().numpy() #_numpy # was just "action" in MATE before, revert?
                rewards[t] = reward
                log_probs[t] = log_prob.detach().cpu().numpy()
                values[t] = value.detach().cpu().item()
                # print(f"       hor. {t}: state is {state}")
                # print(f"       hor. {t}: act is  {action_greedy.detach().cpu().numpy()}")
                # print(f"       hor. {t}: act is  {action_greedy.detach().cpu().numpy()}")

                state = next_state

            # if (t % 5 == 0):
            #     print(*[a.mem_vec.shape[0] + a.mem_vec_new.shape[0] for a in self.actor.mind])
        # print("Min/max rewards:", rewards.min(), rewards.max())
        # print("values", values)
        value = self.run_critic(state)
        last_value = value.detach().cpu().item()
        # self.tracker.add_value('actions', actions)
        # self.tracker.add_value('weights', weights)
        # self.tracker.add_value('rewards', rewards)
        # self.tracker.add_value('probabilities', probabilities)
        if self.states_logging and not self.actor.eval_mode:
            self.tracker.add_value('states', dict((f"{k}", v) for k, v in self.actor.log_states.items()))
            self.actor.log_states = dict((k, []) for k in range(self.message_iterations + 1))
            self.tracker.add_value('actions_gr', {"actions_gr": self.log_actions_gr})
            self.log_actions_gr = []
            if self.actor.memory_used and not self.actor.training:
                self.tracker.add_value('actions_gr_true', {"actions_gr_true": self.log_actions_gr_true})
                self.log_actions_gr_true = []
        if self.actor.memory_used and not self.actor.eval_mode:
            self.tracker.add_value('messages_agents', {"messages_agents": self.actor.log_message_iterations_done_agents})
            self.actor.log_message_iterations_done_agents = []
        return states, actions, rewards, log_probs, values, last_value

    def run_step(self, topo, flows, horizons, iteration = 0, train = False, hash_weights = None, topo_changed=False):
        # calculate weights for given topo and flows (1 exp iteration)
        # TODO: training on the fly
        # adds horizons + 1 values to _all_horizons_phi values
        self.current_flows = flows
        self.current_topology = topo
        self.env.path_calculator.hash_paths = {}

        self.new_topo = topo_changed

        print(f'Episode {self.episode}...')
        states, actions, rewards, log_probs, values, last_value = self.run_episode(True, horizons, hash_weights=hash_weights, keep_memory=True, select_max=True) # was False in MATE before
        returns, advantages = self.gae_estimation(rewards, values, last_value)



        # if self.n_without_update > 0:
        #     self.topo_dict[str(self.episode)] = self.env.G
        #     for i in range(self.horizon):
        #         self.samples.append((str(self.episode), (states[i], actions[i], returns[i], advantages[i], log_probs[i])))
        #     if (self.episode + 1) % self.n_without_update == 0:
        #         actor_losses, critic_losses, losses = self.run_update(self.samples)
        #         self.samples.clear()
        #         self.topo_dict.clear()
        #         self.actor.update_graph(self.env.G)
        #         self.critic.update_graph(self.env.G)

        comm_ratio = self.actor.log_message_iterations_done / self.actor.log_message_iterations_possible \
            if self.actor.log_message_iterations_possible > 0 else 1.0
        print(f'phi = {self.phi(self.env.G, eval=True)}, '
              f'message_iterations done {self.actor.log_message_iterations_done} / {self.actor.log_message_iterations_possible} '
              f'({comm_ratio * 100:.2f} %)')
        PhiCalculator._message_iterations_done.append(self.actor.log_message_iterations_done)
        PhiCalculator._message_iterations_possible.append(self.actor.log_message_iterations_possible)
        PhiCalculator.end_episode()
        self.tracker.end_iteration()
        self.env.end_iteration()
        # PhiCalculator.plot_result()
        self.episode += 1
        return self.env.hash_weights

    def run_update(self, samples):
        # update weights based on given trajectories
        actor_losses, critic_losses, losses = [], [], []
        inds = np.arange(self.horizon * self.n_without_update)
        for _ in range(self.epochs):
            np.random.shuffle(inds)
            for start in range(0, self.horizon * self.n_without_update, self.batch_size):
                end = start + self.batch_size
                minibatch_ind = inds[start:end]
                self.optimizer.zero_grad()
                minibatch_topo_dict = dict()
                for i in minibatch_ind:
                    # state, action, reward, adv, log_prob = samples[i][1]
                    if samples[i][0] in minibatch_topo_dict:
                        minibatch_topo_dict[samples[i][0]].append(samples[i][1])
                    else:
                        minibatch_topo_dict[samples[i][0]] = [samples[i][1]]
                actor_loss, critic_loss, loss = self.compute_losses_and_grads(minibatch_topo_dict)
                # print("\n-----------------------\n")
                # print(f"loss: {loss}")
                # print("-----------------------")
                # print("actor loss")
                # print(actor_loss.detach().numpy())
                # check_gradients(self.actor)
                # print("-----------------------")
                # print("critic loss")
                # print(critic_loss.detach().numpy())
                # check_gradients(self.critic)
                # print("\n-----------------------\n")

                self.optimizer.step()
                actor_losses.append(actor_loss.detach().cpu().numpy())
                critic_losses.append(critic_loss.detach().cpu().numpy())
                losses.append(loss.detach().cpu().numpy())
        return actor_losses, critic_losses, losses

    def train_and_evaluate(self, path):
        # train the model
        torch.enable_grad()
        training_episode = 0
        self.set_dataset(path)
        new_sample = False

        samples = list() # [(topo_name, (states : 1DTensor, action : 1DTensor, reward, adv, log_probs: 1DTensor)), ...]

        # states_cur = np.zeros((self.n_without_update * self.horizon, self.env.n_links *
        #                    self.actor.num_features), dtype=np.float32)
        # actions_cur = np.empty(self.n_without_update * self.horizon, dtype=np.float32)
        # log_probs_cur = np.empty(self.n_without_update * self.horizon, dtype=np.float32)
        # advantages_cur = np.empty(self.n_without_update * self.horizon, dtype=np.float32)
        # returns_cur = np.empty(self.n_without_update * self.horizon, dtype=np.float32)
        # n_msg_iter_cur = np.empty(self.n_without_update * self.horizon, dtype=np.float32)

        eval_done = 0

        while training_episode < self.episodes:
            upd_ep = training_episode % self.n_without_update \
                if self.n_without_update > 0 else training_episode
            if training_episode % self.change_sample_period == 0:
                self.update_sample(training_episode)
                new_sample = True

            print('Episode ', training_episode, '...')
            if eval_done == 1:
                self.new_topo = True
                eval_done = 0

            if self.check_mem_and_time:
                write_memory_usage()

            with torch.no_grad():
                states, actions, rewards, log_probs, values, last_value = self.run_episode(new_sample=new_sample, keep_memory=True) # keep_memory was False in MATE before
                # print(f"Rewards: mean={rewards.mean()}, std={rewards.std()}")
            new_sample = False

            if self.strategy != "EQUAL":
                returns, advantages = self.gae_estimation(rewards, values, last_value)

                if self.current_gml not in self.topo_dict:
                    self.topo_dict[self.current_gml] = self.env.G
                for i in range(self.horizon):
                    samples.append((self.current_gml, (states[i], actions[i], returns[i], advantages[i], log_probs[i])))
                # states_cur[upd_ep*self.horizon : (upd_ep+1)*self.horizon] = states
                # actions_cur[upd_ep*self.horizon : (upd_ep+1)*self.horizon] = actions
                # returns_cur[upd_ep*self.horizon : (upd_ep+1)*self.horizon] = returns
                # advantages_cur[upd_ep*self.horizon : (upd_ep+1)*self.horizon] = advantages
                # log_probs_cur[upd_ep*self.horizon : (upd_ep+1)*self.horizon] = log_probs
                # n_msg_iter_cur[upd_ep*self.horizon : (upd_ep+1)*self.horizon] = n_msg_iter

                # print(f"Advantages: mean={advantages.mean()}, std={advantages.std()}")
                # print(f"Returns: mean={returns.mean()}, max={returns.max()}")
                # print(f"Values: mean={values.mean()}, max={values.max()}")
                if upd_ep + 1 == self.n_without_update:
                    gml = self.current_gml
                    actor_losses, critic_losses, losses = self.run_update(samples)
                    samples.clear()
                    if gml != self.current_gml:
                        self.current_gml = gml
                        self.new_topo = True
                        new_sample = True
                        self.actor.update_graph(self.env.G)
                        self.critic.update_graph(self.env.G)
                # actor_losses, critic_losses, losses = self.run_update(states, actions, returns,
                #                                                 advantages, log_probs)

            comm_ratio = self.actor.log_message_iterations_done / self.actor.log_message_iterations_possible \
                if self.actor.log_message_iterations_possible > 0 else 1.0
            print(f'phi = {self.phi(self.env.G)}, '
                  f'message_iterations done {self.actor.log_message_iterations_done} / {self.actor.log_message_iterations_possible} '
                  f'({comm_ratio * 100:.2f} %)')
            PhiCalculator._message_iterations_done.append(self.actor.log_message_iterations_done)
            PhiCalculator._message_iterations_possible.append(self.actor.log_message_iterations_possible)

            if self.save_checkpoints and (training_episode + 1) % 1000 == 0:
                    # self.actor._set_inputs(states[0])
                    # self.critic._set_inputs(states[0])
                    self.save_model(self.checkpoint_dir, training_episode + 1)

            if (training_episode + 1) % self.eval_period == 0 or training_episode == 0:

                print('eval')
                topo, flows = self.current_topology, self.current_flows # TODO: eval env
                gml = self.current_gml
                self.env.eval_mode = True # don't memorize phi values
                self.actor.update_memory = False
                self.actor.eval_mode = True
                value, min_value, eval_points_full, eval_inputs, \
                    eval_msg_done, eval_msg_possible = self.eval_on_dataset()
                self.actor.update_memory = self.actor.cfg.update_memory
                self.actor.eval_mode = False
                if self.current_gml != gml:
                    self.new_topo = True
                eval_done = 1
                self.env.eval_mode = False
                self.current_topology, self.current_flows = topo, flows
                self.current_gml = gml
                new_sample = True
                PhiCalculator._eval_values.append(value)
                PhiCalculator._eval_points.append(training_episode)
                PhiCalculator._eval_values_full.append(eval_points_full)
                PhiCalculator._eval_msg_done.append(eval_msg_done)
                PhiCalculator._eval_msg_possible.append(eval_msg_possible)
                if len(PhiCalculator._eval_inputs) == 0:
                    PhiCalculator._eval_inputs.append(eval_inputs)
                # if self.change_traffic and self.eval_episode % self.change_sample_period == 0:
                #     self.change_sample = True


            PhiCalculator.end_episode()
            self.tracker.end_iteration()
            self.env.end_iteration()

            if training_episode > 0 and ((training_episode + 1) % self.plot_period == 0):
                PhiCalculator.plot_result()
            training_episode += 1
            if True:
                continue
            if training_episode % 50 == 0 or training_episode == 10 or training_episode == 25:
                with open("weights_output.txt", "+a") as f:
                    ws = []
                    bs = []
                    for layer_i in range(len(self.actor.layers)):
                        ws.append(self.actor.layers[layer_i].get_weights()[0])
                        bs.append(self.actor.layers[layer_i].get_weights()[1])
                        for i in range(len(ws)):
                            f.write(f"\nactor training_episode {training_episode}, {np.shape(ws[i])} weights:")
                            f.write(str(ws[i]))
                            f.write(f"\nactor training_episode {training_episode}, {np.shape(bs[i])} biases:")
                            f.write(str(bs[i]))
                    ws = []
                    bs = []
                    for layer_i in range(len(self.critic.layers)):
                        ws.append(self.critic.layers[layer_i].get_weights()[0])
                        bs.append(self.critic.layers[layer_i].get_weights()[1])
                        for i in range(len(ws)):
                            f.write(f"\ncritic training_episode {training_episode}, {np.shape(ws[i])} weights:")
                            f.write(str(ws[i]))
                            f.write(f"\ncritic training_episode {training_episode}, {np.shape(bs[i])} biases:")
                            f.write(str(bs[i]))
                    f.write("\n\n")

        # save memory
        if self.actor.memory_used and self.actor.update_memory:
            print("saving memory...")
            memory_layers = []
            for layer in range(self.actor.message_iterations):
                mem = self.actor.mind[layer].mem_vec # 0th memory layer, (mem_size, n_agents, 16)
                act = self.actor.mind[layer].act_vec # (mem_size, n_agents, 16)
                acts = np.empty([act.shape[0], act.shape[1], 1], dtype=np.float32)
                state_to_act = np.empty([act.shape[0], act.shape[1], mem.shape[2] + 1])
                for i in range(act.shape[0]):
                    policy = self.actor.readout(torch.tensor(act[i, :, :], requires_grad=False).type(torch.float32).to(self.device))
                    logits = torch.reshape(policy, (-1,))
                    logits_reshaped = torch.reshape(logits, shape=(torch.numel(logits) // self.num_actions, self.num_actions))
                    # dist = torch.distributions.Categorical(logits=logits_reshaped)
                    action = torch.argmax(logits_reshaped, dim=1).to(torch.int32)
                    acts[i] = action.cpu().numpy()[:, None]
                state_to_act = np.concatenate((mem, act, acts), axis=2)
                memory_layers.append(state_to_act)
            np.savez(os.path.join(self.checkpoint_dir, f'memory.npz'), *memory_layers)

        self.env.get_hash_weights()
        return self.env.hash_weights

    def calc_actions(self, path):
        # torch.enable_grad()
        training_episode = 0
        self.set_dataset(path)
        new_sample = False

        eval_done = 0
        states_set, actions_set = load_states(self.states_path, load_actions=True)
        n_iterations = len(states_set.keys()) - 1
        n_episodes = states_set[0].shape[0]
        n_horizons = states_set[0].shape[1]
        n_agents = states_set[0].shape[2]
        link_state_size = states_set[1].shape[-1]
        link_state_size_0 = states_set[0].shape[-1]

        with torch.no_grad():
            for episode in range(n_episodes):
                if training_episode % self.change_sample_period == 0:
                    self.update_sample(training_episode)
                    new_sample = True

                print('Episode ', training_episode, '...')
                if eval_done == 1:
                    self.new_topo = True
                    eval_done = 0

                if self.check_mem_and_time:
                    write_memory_usage()

                # run one episode
                horizons = self.horizon
                keep_memory = True
                hash_weights = None
                select_max = False
                if new_sample:
                    self.env.path_calculator.gml = self.current_gml
                    self.env.reset(self.current_topology, self.current_flows, hash_weights=hash_weights, new_topo=self.new_topo, equal=select_max)
                    if keep_memory == False or self.new_topo:
                        self.actor.update_graph(self.env.G, keep_memory=keep_memory)
                        self.critic.update_graph(self.env.G)
                else:
                    self.env.reset()
                # states = np.zeros((horizons, self.env.n_links *
                                   # self.actor.num_features), dtype=np.float32)
                # actions = np.zeros([horizons, self.env.G.number_of_edges()], dtype=np.float32)

                self.episode_min_phi = 10.0

                # print('Initial phi:', self.phi(self.env.G, eval=True))
                for t in range(n_horizons):
                    state = states_set[0][episode, t].T.flatten()
                    action, log_prob, raw_probs, action_greedy, action_greedy_true = self.act(state, select_max=select_max)
                    if self.states_logging and not self.actor.eval_mode:
                        self.log_actions_gr.append(action_greedy.detach().cpu().numpy())
                    # states[t] = state
                    # actions[t] = action.detach().numpy()
                    print(f"ep. {episode}, hor. {t}: state is {state}")
                    # print(f"ep. {episode}, hor. {t}: act is  {action_greedy.detach().cpu().numpy()}")

                    # print(f"agent 20 state[0] += ______: {action_greedy.detach().cpu().numpy()}")
                    act_orig = action_greedy.detach().cpu().numpy()
                    n_aug = 10
                    d = 0.05
                    for agent_i in range(n_agents):
                        for state_el_i in range(2):
                            for sgn in (1, -1):
                                change = 0
                                act_cur = act_orig
                                state = states_set[0][episode, t].T.flatten()
                                print(f"           without changes: {act_orig}")
                                for aug_i in range(n_aug):
                                    change += d
                                    state[agent_i + n_agents * state_el_i] += sgn * d
                                    action, log_prob, raw_probs, action_greedy, action_greedy_true = self.act(state, select_max=select_max)
                                    act = action_greedy.detach().cpu().numpy()
                                    if self.states_logging and not self.actor.eval_mode:
                                        self.log_actions_gr.append(act)
                                    if not np.array_equal(act, act_cur) or aug_i == n_aug - 1:
                                        print(f"agent {agent_i:2} state[{state_el_i}] += {sgn * change:5.2f}: {act}")
                                        # break
                                    act_cur = act
                                print()

                    # print(f"ep. {episode}, hor. {t}: act was {actions_set[episode, t]}")

                # print(f"{self.actor.eval_mode=}")
                if self.states_logging and not self.actor.eval_mode:
                    # print(f"tracker adding: states ({len(self.actor.log_states)}), actions_gr ({len(self.log_actions_gr)})")
                    self.tracker.add_value('states', dict((f"{k}", v) for k, v in self.actor.log_states.items()))
                    self.actor.log_states = dict((k, []) for k in range(self.message_iterations + 1))
                    self.tracker.add_value('actions_gr', {"actions_gr": self.log_actions_gr})
                    self.log_actions_gr = []
                new_sample = False

                if self.current_gml not in self.topo_dict:
                    self.topo_dict[self.current_gml] = self.env.G

                PhiCalculator._message_iterations_done.append(self.actor.log_message_iterations_done)
                PhiCalculator._message_iterations_possible.append(self.actor.log_message_iterations_possible)

                PhiCalculator.end_episode()
                self.tracker.end_iteration()
                self.env.end_iteration()

                if training_episode > 0 and ((training_episode + 1) % self.plot_period == 0):
                    PhiCalculator.plot_result()
                training_episode += 1

        # self.env.get_hash_weights()
        # return self.env.hash_weights
        return None

    def generate_eval_env(self, current_flows, current_topology, hash_function):
        self.eval_envs = {}
        for eval_env_type in self.eval_env_type:
            self.eval_envs[eval_env_type] = Environment(env_type=eval_env_type,
                                                        traffic_profile=self.env.traffic_profile,
                                                        routing=self.env.routing,
                                                        current_flows=current_flows,
                                                        current_topology=current_topology,
                                                        hash_function=hash_function,
                                                        action_config=self.action_config)

    def generate_eval_actor_critic_functions(self):
        self.eval_actor = {}
        self.eval_critic = {}

        for eval_env_type in self.eval_env_type:
            self.eval_actor[eval_env_type] = Actor(
                self.eval_envs[eval_env_type].G, self.env.get_adj_matrix(), num_features=self.env.num_features).to(self.device)

            self.eval_critic[eval_env_type] = Critic(self.eval_envs[eval_env_type].G,
                                                     self.env.get_adj_matrix(),
                                                     num_features=self.env.num_features).to(self.device)

    def update_eval_actor_critic_functions(self):
        for eval_env_type in self.eval_env_type:
            # actor
            for w_model, w_eval_actor in zip(self.actor.parameters(),
                                             self.eval_actor[eval_env_type].parameters()):
                w_eval_actor.data.copy_(w_model.data)

            # critic
            for w_model, w_eval_critic in zip(self.critic.parameters(),
                                              self.eval_critic[eval_env_type].parameters()):
                w_eval_critic.data.copy_(w_model.data)

    def eval_on_dataset(self):
        print('running evaluation')
        results = []
        min_results = []
        inputs = []
        msg_done_values = []
        msg_possible_values = []
        for x in self.test_set:
            if x.replace('test', 'train') in self.train_set:
                topo_dir = x.replace('test', 'train')
            else:
                topo_dir = x
            tm_dir = x
            tm_dir_stem = os.path.split(tm_dir)[1]
            for y in self.test_set[x]:
                print('eval sample', x, y)
                topo, flows = self.load_sample(topo_dir, tm_dir, y)
                res, msg_done, msg_possible = self.eval_on_sample(topo, flows)
                results.append(res)
                min_results.append(self.episode_min_phi)
                inputs.append(os.path.join(tm_dir_stem, y))
                msg_done_values.append(msg_done)
                msg_possible_values.append(msg_possible)
        ans = np.mean(results)
        min_ans = np.mean(min_results)
        msg_done_sum = np.sum(msg_done_values)
        msg_possible_sum = np.sum(msg_possible_values)
        comm_ratio = msg_done_sum / msg_possible_sum if msg_possible_sum > 0 else 1.0
        print(f'eval final values: {ans}')
        print(f'eval min values: {min_ans}')
        print(f'eval message iterations done {msg_done_sum} / {msg_possible_sum} ({comm_ratio * 100:.2f} %)')
        return ans, min_ans, results, inputs, msg_done_values, msg_possible_values

    def eval_on_sample(self, topo, flows):
        results = []
        self.current_topology, self.current_flows = topo, flows
        msg_iter_done_before = self.actor.log_message_iterations_done
        msg_iter_possible_before = self.actor.log_message_iterations_possible
        for j in range(self.num_eval_samples): # 1 iter
            with torch.no_grad():
                states, actions, rewards, log_probs, values, last_value = self.run_episode(True, select_max=True, keep_memory=True) # keep_memory was False in MATE before
            # reward = np.sum(rewards)
            phi = self.phi(self.env.G, eval = True)
            results.append(phi)
        msg_iter_done = self.actor.log_message_iterations_done - msg_iter_done_before
        msg_iter_possible = self.actor.log_message_iterations_possible - msg_iter_possible_before
        return np.mean(results), msg_iter_done, msg_iter_possible


    def compute_actor_loss(self, new_log_probs, old_log_probs, advantages):
        ratio = torch.exp(new_log_probs - torch.tensor(old_log_probs, device=new_log_probs.device).detach()) # (25, n_links)
        adv_t = torch.tensor(advantages, requires_grad = True, device=new_log_probs.device)
        pg_loss_1 = torch.func.vmap(lambda x:-x[0] * x[1])((adv_t, ratio))
        pg_loss_2 = torch.func.vmap(lambda x: -x[0] * torch.clamp(x[1], 1.0 - self.clip_param, 1.0 + self.clip_param))((adv_t, ratio))
        # pg_loss shape = (25, n_links)
        actor_loss = torch.mean(torch.maximum(pg_loss_1, pg_loss_2), dim = 1)
        return actor_loss # vector of mean losses for each horizon

    def get_new_log_prob_and_entropy(self, state, action):
        self.actor.train()
        logits, _ = self.actor(state)
        logits_reshaped = torch.reshape(logits, shape=(torch.numel(logits) // self.num_actions, self.num_actions))

        dist = torch.distributions.Categorical(logits=logits_reshaped)

        log_probs = dist.log_prob(action)
        entropy = dist.entropy()

        return log_probs, entropy

    def compute_losses_and_grads(self, minibatch_dict : dict):
        topos = list(minibatch_dict.items())
        advantages = np.array([])
        entropy_losses = torch.tensor([], requires_grad=True, device=self.device)
        critic_losses = torch.tensor([], requires_grad=True, device=self.device)
        actor_losses = torch.tensor([], requires_grad=True, device=self.device)
        for x in topos:
            gml = x[0]
            if gml != self.current_gml:
                self.actor.update_graph(self.topo_dict[gml])
                self.critic.update_graph(self.topo_dict[gml])
                self.current_gml = gml
                self.new_topo = True
            states, actions, returns, adv, log_probs = zip(*x[1])

            states_t = torch.tensor(np.array(states), requires_grad = True, device=self.device) # (25, 16)
            actions_t = torch.tensor(np.array(actions), requires_grad = True, device=self.device) # (25, A)
            old_log_probs = np.array(log_probs)
            advantages = adv
            new_log_probs, entropy = torch.func.vmap(lambda x: self.get_new_log_prob_and_entropy(x[0], x[1]),
                                                        randomness='different')((states_t, actions_t))


            self.critic.train()
            values_ = torch.func.vmap(lambda x: self.critic(x), randomness='different')(states_t)
            values_ = values_.reshape(-1)

            returns_t_ = torch.tensor(np.array(returns), requires_grad=True, device=self.device)
            critic_losses = torch.cat((critic_losses, torch.square(returns_t_ - values_)))
            entropy_losses = torch.cat((entropy_losses, torch.mean(entropy, dim=1))) # entropy shape = (25, n_links)
            actor_losses = torch.cat((actor_losses, self.compute_actor_loss(new_log_probs, old_log_probs, advantages)))


        critic_loss = torch.mean(critic_losses)
        entropy_loss = torch.mean(entropy_losses)
        actor_loss = torch.mean(actor_losses)
        loss = actor_loss - self.entropy_loss_factor * entropy_loss + self.critic_loss_factor * critic_loss

        loss.backward()
        if self.max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=self.max_grad_norm)
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=self.max_grad_norm)

        return actor_loss, critic_loss, loss


    def act(self, state, select_max=False):
        self.actor.eval()
        logits, logits_true = self.actor(torch.tensor(state, requires_grad = False, device=self.device))

        logits_reshaped = torch.reshape(logits, shape=(torch.numel(logits) // self.num_actions, self.num_actions))

        dist = torch.distributions.Categorical(logits=logits_reshaped)

        greedy_actions = torch.argmax(logits_reshaped, dim=1).to(torch.int32)
        sampled_actions = dist.sample()

        random_val = torch.rand((), device=self.device)
        if select_max:
            action = greedy_actions
        else:
            action = torch.where(random_val < self.eps, sampled_actions, greedy_actions)
        log_probs = dist.log_prob(action)

        if logits_true is not None:
            logits_true_reshaped = torch.reshape(logits_true, shape=(torch.numel(logits_true) // self.num_actions, self.num_actions))
            greedy_actions_true = torch.argmax(logits_true_reshaped, dim=1).to(torch.int32)
        else:
            greedy_actions_true = None
        return action, log_probs, None, greedy_actions, greedy_actions_true


    def eval_act(self, actor, state, select_max=False):
        self.actor.eval()
        logits, _ = self.actor(state)
        logits_reshaped = torch.reshape(logits, shape=(torch.numel(logits) // self.num_actions, self.num_actions))
        probs = [torch.distributions.Categorical(logits=t) for t in logits_reshaped]
        if select_max:
            action = torch.argmax(logits_reshaped, dim=1)

        else:
            action = torch.stack([p.sample() for p in probs])

        log_probs = torch.stack([p.log_prob(a) for (a, p) in zip(action, probs)])

        return action, log_probs


    def run_critic(self, state):
        self.critic.eval()
        return self.critic(torch.tensor(state, requires_grad = False, device=self.device))

    def save_model(self, checkpoint_dir, episode):
        ok = False
        for _ in range(N_IO_TRIES):
            try:
                torch.save(self.actor.state_dict(), checkpoint_dir + f'/{episode}_actor.pt', pickle_module=dill)
                torch.save(self.critic.state_dict(), checkpoint_dir + f'/{episode}_critic.pt', pickle_module=dill)
                ok = True
                break
            except Exception as e:
                print(f"IOERROR: {e}")
                sleep(5)
        if ok == False:
            raise Exception("couldn't save model due to repeated IOERROR")

    def load_saved_model(self, model_dir, only_eval):
        # for checking need new models, model is in .pt file

        if os.path.isfile(os.path.join(model_dir, 'actor_oldformat.pt')):
            # load actor
            actor_model = torch.load(os.path.join(model_dir, 'actor_oldformat.pt'), pickle_module=dill,
                                     map_location=self.device)
            for w_model, w_actor in zip(actor_model.parameters(),
                                         self.actor.parameters()):
                w_actor.data.copy_(w_model.data)

            # load critic
            if not only_eval:
                critic_model = torch.load(os.path.join(model_dir, 'critic_oldformat.pt'), pickle_module=dill,
                                          map_location=self.device)
                for w_model, w_critic in zip(critic_model.parameters(),
                                         self.critic.parameters()):
                    w_critic.data.copy_(w_model.data)
        else:
            self.actor.load_state_dict(torch.load(os.path.join(model_dir, 'actor.pt'), weights_only=True, map_location=self.device))
            self.critic.load_state_dict(torch.load(os.path.join(model_dir, 'critic.pt'), weights_only=True, map_location=self.device))

        self.model_dir = model_dir
        self.reload_model = True


    def write_eval_results(self, step, value):
        csv_dir = os.path.join('./notebooks/logs', self.experiment_identifier)
        if not os.path.exists(csv_dir):
            os.makedirs(csv_dir)
        with open(csv_dir + '/results.csv', "a") as csv_file:
            writer = csv.writer(csv_file, delimiter=',')
            writer.writerow([step, value])

    def set_experiment_identifier(self, only_eval):
        self.only_eval = only_eval
        mode = 'eval' if only_eval else 'training'

        if mode == 'training':
            # PPOAGENT
            batch = 'batch' + str(self.batch_size)
            gae_lambda = 'gae' + str(self.gae_lambda)
            lr = 'lr' + str(self.optimizer.param_groups[0]['lr'])
            epsilon = 'epsilon' + str(self.optimizer.param_groups[0]['eps'])
            clip = 'clip' + str(self.clip_param)
            gamma = 'gamma' + str(self.gamma)
            episodes = 'episodes' + str(self.eval_period)
            horizon = 'horizons' + str(self.horizon)
            epoch = 'epoch' + str(self.epochs)
            greedy_eps = 'greedyeps' + str(self.eps)
            agent_folder = '-'.join((batch, lr, epsilon, gae_lambda, clip, gamma,
                                     episodes, horizon, epoch, greedy_eps))

            # ACTOR-CRITIC-ENV
            state_size = 'size' + str(self.actor.link_state_size)
            iters = 'iters' + str(self.actor.message_iterations)
            aggregation = self.actor.aggregation
            nn_size = 'nnsize' + str(self.actor.final_hidden_layer_size)
            dropout = 'drop' + str(self.actor.dropout_rate)
            activation = self.actor.activation_fn
            base_reward = self.env.base_reward
            reward_comp = self.env.reward_computation
            function_folder = '-'.join((state_size, iters, aggregation, nn_size, dropout, activation,
                                        base_reward, reward_comp))

            self.experiment_identifier = os.path.join(agent_folder, function_folder)

        else:
            model_dir = self.model_dir

            network = '+'.join([str(elem) for elem in self.env.env_type])
            traffic_profile = self.env.traffic_profile
            routing = self.env.routing
            eval_env_folder = ('-').join([network, traffic_profile, routing])

            # RELOADED MODEL
            env_folder = os.path.join(model_dir.split('/')[3])
            agent_folder = os.path.join(model_dir.split('/')[4])
            function_folder = os.path.join(model_dir.split('/')[5])
            episode = os.path.join(model_dir.split('/')[6])

            self.experiment_identifier = os.path.join(mode, eval_env_folder, env_folder, agent_folder, function_folder,
                                                      episode)
        return self.experiment_identifier
