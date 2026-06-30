import numpy as np
import torch
from dte_stand.algorithm.mate2l.lib.metrics import *
from dte_stand.algorithm.mate2l.config import ActorCfg, SaActorCfg

from time import sleep
from memory_profiler import profile


def get_activation_method(activation_str):
    if activation_str == 'relu':
        return torch.nn.ReLU()
    elif activation_str == 'sigmoid':
        return torch.nn.Sigmoid()
    elif activation_str == 'tanh':
        return torch.nn.Tanh()
    elif activation_str == 'linear':
        return torch.nn.Identity()
    elif activation_str == 'softplus':
        return torch.nn.Softplus()
    else:
        return lambda x : 0

class CreateMessage(torch.nn.Module):
    def __init__(self, input_size, hidden_layer_size, link_state_size, initializer, init_gain, activation_fn):
        super().__init__()
        # layers
        self.nn1 = torch.nn.Linear(input_size, hidden_layer_size)
        self.nn2 = torch.nn.Linear(hidden_layer_size, link_state_size)
        self.act1 = get_activation_method(activation_fn)
        self.act2 = get_activation_method(activation_fn)

        # init
        self.initializer = initializer
        self.init_gain = init_gain

        # Initialization
        self.initializer(self.nn1.weight, gain= self.init_gain)
        self.initializer(self.nn2.weight, gain= self.init_gain)
        torch.nn.init.zeros_(self.nn1.bias)
        torch.nn.init.zeros_(self.nn2.bias)

    def forward(self, x):
        x = self.nn1(x)
        x = self.act1(x)
        x = self.nn2(x)
        x = self.act2(x)
        return x


class LinkUpdate(torch.nn.Module):
    def __init__(self, input_size, first_hidden_layer_size, final_hidden_layer_size, link_state_size,
                 initializer, init_gain, activation_fn):
        super().__init__()
        # layers
        self.nn1 = torch.nn.Linear(input_size, first_hidden_layer_size)
        self.nn2 = torch.nn.Linear(first_hidden_layer_size, final_hidden_layer_size)
        self.nn3 = torch.nn.Linear(final_hidden_layer_size, link_state_size)
        self.act1 = get_activation_method(activation_fn)
        self.act2 = get_activation_method(activation_fn)
        self.act3 = get_activation_method(activation_fn)

        # init
        self.initializer = initializer
        self.init_gain = init_gain

        # Initialization
        self.initializer(self.nn1.weight, gain= self.init_gain)
        self.initializer(self.nn2.weight, gain= self.init_gain)
        self.initializer(self.nn3.weight, gain= self.init_gain)
        torch.nn.init.zeros_(self.nn1.bias)
        torch.nn.init.zeros_(self.nn2.bias)
        torch.nn.init.zeros_(self.nn3.bias)

    def forward(self, x):
        x = self.nn1(x)
        x = self.act1(x)
        x = self.nn2(x)
        x = self.act2(x)
        x = self.nn3(x)
        x = self.act3(x)
        return x

class ReadOut(torch.nn.Module):
    def __init__(self, input_size, first_hidden_layer_size, final_hidden_layer_size, num_actions,
                 initializer1, init_gain1, initializer2, init_gain2,
                 activation_fn, final_activation_fn, regularizer_name = None, regularizer_rate = 0.0,
                 drop_rate = 0.0):
        super().__init__()
        # layers
        self.nn1 = torch.nn.Linear(input_size, first_hidden_layer_size)
        self.drop1 = torch.nn.Dropout(drop_rate)
        self.nn2 = torch.nn.Linear(first_hidden_layer_size, final_hidden_layer_size)
        self.drop2 = torch.nn.Dropout(drop_rate)
        self.nn3 = torch.nn.Linear(final_hidden_layer_size, num_actions)
        self.act1 = get_activation_method(activation_fn)
        self.act2 = get_activation_method(activation_fn)
        self.act3 = get_activation_method(final_activation_fn)

        # init
        self.initializer1 = initializer1
        self.init_gain1 = init_gain1
        self.initializer2 = initializer2
        self.init_gain2 = init_gain2


        # Regularizer (TODO IN FUTURE)
        self.reg_rate = regularizer_rate
        self.reg = self.get_regularizer(regularizer_name, self.reg_rate)

        # Initialization
        self.initializer1(self.nn1.weight, self.init_gain1)
        self.initializer1(self.nn2.weight, self.init_gain1)
        self.initializer2(self.nn3.weight, self.init_gain2)
        torch.nn.init.zeros_(self.nn1.bias)
        torch.nn.init.zeros_(self.nn2.bias)
        torch.nn.init.zeros_(self.nn3.bias)

    def get_regularizer(self, reg_name, reg_rate):
        if reg_name == None:
            return self.no_regularizer

    def no_regularizer(self):
        return 0

    def forward(self, x):
        x = self.nn1(x)
        x = self.act1(x)
        x = self.drop1(x)
        x = self.nn2(x)
        x = self.act2(x)
        x = self.drop2(x)
        x = self.nn3(x)
        x = self.act3(x)
        return x


"""
This version doesn't use memory while learning. It is done to solve problem
with using vmap (cant use .numpy() to BatchedTensors)
"""
class Actor(torch.nn.Module):
    def __init__(self,
                 cfg: SaActorCfg,
                 graph,
                 adj_matrix,
                 num_actions=1,
                 num_features=2,
                 link_state_size=16,
                 aggregation='min_max',
                 first_hidden_layer_size=128,
                 dropout_rate=0.15,
                 final_hidden_layer_size=64,
                 message_iterations=8,
                 activation_fn='tanh',
                 final_activation_fn='linear',
                 memory_state=None,
                 device=torch.device('cpu')):

        super(Actor, self).__init__()

        # actor-specific configuration
        self.cfg = cfg

        self.mode = cfg.mpnn_mode
        self.memory_lookup = cfg.memory_lookup
        self.networks = cfg.networks
        self.update_memory = cfg.update_memory
        self.eval_mode = False

        self.device = device

        # HYPERPARAMETERS
        self.num_actions = num_actions
        self.num_features = num_features
        self.link_state_size = link_state_size
        self.message_hidden_layer_size = final_hidden_layer_size
        self.aggregation = aggregation
        self.message_iterations = message_iterations

        # FIXED INPUTS
        if graph:
            self.n_links = graph.number_of_edges()
            self.incoming_links = torch.tensor(graph.nodes()['graph_data']['incoming_links'], device=self.device)
            self.outcoming_links = torch.tensor(graph.nodes()['graph_data']['outcoming_links'], device=self.device)
            self.incoming_links_2 = torch.tensor(graph.nodes()['graph_data']['il2'], device=self.device)
            self.outcoming_links_2 = torch.tensor(graph.nodes()['graph_data']['ol2'], device=self.device)
            self.opposite_links = torch.tensor(graph.nodes()['graph_data']['opposites'], device=self.device)

        # GAT
        self.gat_used = self.cfg.use_gat
        self.num_heads = self.cfg.gat_num_heads
        if adj_matrix:
            self.adj_matrix = adj_matrix

        self.prev_edges = None
        self.all_edges = None

        # NEURAL NETWORKS
        self.hidden_layer_initializer = torch.nn.init.orthogonal_
        self.hidden_layer_initializer_gain = np.sqrt(2)
        self.final_layer_initializer = torch.nn.init.orthogonal_
        self.final_layer_initializer_gain = 0.01
        self.kernel_regularizer = None
        self.kernel_regularizer_rate = 0.0
        self.activation_fn = activation_fn
        self.final_hidden_layer_size = final_hidden_layer_size
        self.first_hidden_layer_size = first_hidden_layer_size
        self.dropout_rate = dropout_rate
        self.final_activation_fn = final_activation_fn

        self.message_input_shape = 2 * self.link_state_size
        if self.aggregation == 'sum':
            self.link_input_shape = 2 * self.link_state_size
        elif self.aggregation == 'min_max':
            self.link_input_shape_1 = 3 * self.link_state_size
            self.link_input_shape_2 = 5 * self.link_state_size if self.mode == '1' or self.mode == '2' else 3 * self.link_state_size
            self.link_input_shape_3 = 3 * self.link_state_size
        self.readout_input_shape = self.link_state_size

        if self.mode == '4' or self.mode == '5':
            self.first_iter_shape = self.link_state_size
            self.second_iter_shape = self.link_state_size
        elif self.memory_lookup == 'state':
            self.first_iter_shape = 2
            self.second_iter_shape = self.link_state_size
        else:
            self.first_iter_shape = 3 * 2
            self.second_iter_shape = 3 * self.link_state_size
        self.define_network()

        self.init_memory_state = memory_state

        # MEMORY
        self.memory_used = self.cfg.use_memory
        if graph and self.memory_used and self.message_iterations > 0:
            self.init_memory()

        # MEMORY LOG
        self.log_message_iterations_done = 0
        self.log_message_iterations_possible = 0
        self.states_logging = self.cfg.states_logging
        if self.states_logging:
            self.log_states = dict((k, []) for k in range(self.message_iterations + 1))
        if self.memory_used:
            self.log_message_iterations_done_agents = []

        # Indicates if model can be called inside tf.function
        self.can_compile = not self.memory_used

    def init_memory(self):
        # TODO: GAT
        kwargs = {}
        if self.cfg.memory_type == 'bypass':
            Memory = BypassMemory
        elif self.cfg.memory_type == 'cluster':
            Memory = ClusterMemory
        else:
            Memory = ClusterMemory
        self.memory_size = self.cfg.memory_size
        self.mind = [Memory(0, self.memory_size, self.cfg.threshold, self.n_links, self.first_iter_shape,
            self.num_actions, self.cfg.P, self.cfg.var,
            initial_state=self.init_memory_state['arr_0'] if self.init_memory_state is not None else None, **kwargs)
        ]
        self.mind += [Memory(i, self.memory_size, self.cfg.threshold, self.n_links, self.second_iter_shape,
            self.num_actions, self.cfg.P, self.cfg.var,
            initial_state=self.init_memory_state[f'arr_{i}'] if self.init_memory_state is not None else None, **kwargs)
            for i in range(1, self.message_iterations)
        ]


    def update_message_iterations(self, message_iterations):
        self.message_iterations = message_iterations

    def update_graph(self, graph, keep_memory=False):
        self.n_links = graph.number_of_edges()
        self.incoming_links = torch.tensor(graph.nodes()['graph_data']['incoming_links'], device=self.device)
        self.outcoming_links = torch.tensor(graph.nodes()['graph_data']['outcoming_links'], device=self.device)
        self.incoming_links_2 = torch.tensor(graph.nodes()['graph_data']['il2'], device=self.device)
        self.outcoming_links_2 = torch.tensor(graph.nodes()['graph_data']['ol2'], device=self.device)
        self.opposite_links = torch.tensor(graph.nodes()['graph_data']['opposites'], device=self.device)
        if self.memory_used:
            if keep_memory == False or self.prev_edges is None:
                self.init_memory()
            if keep_memory == True:
                edges = list(graph.edges())
                # if self.prev_edges is not None:
                #     inserts, removes = [], []
                #     j = 0 # pos in prev edges
                #     i = 0 # pos in edges
                #     ins_cnt = 0 # amount of inserts
                #     while i < len(edges):
                #         if j >= len(self.prev_edges):
                #             inserts.append(j)
                #             i += 1
                #             ins_cnt += 1
                #             continue
                #         if edges[i] == self.prev_edges[j]:
                #             i += 1
                #             j += 1
                #             continue
                #         if edges[i] not in self.prev_edges: # new edge
                #             inserts.append(j)
                #             i += 1
                #             ins_cnt += 1
                #             continue
                #         if self.prev_edges[j] not in edges: # removed edge
                #             removes.append(j + ins_cnt)
                #             j += 1
                #     while j < len(self.prev_edges):
                #         removes.append(j + ins_cnt)
                #         j += 1
                #     for k in range(len(self.mind)):
                #         self.mind[k].topology_changes(inserts, removes, len(edges))
                if self.all_edges is not None:
                    indices = []
                    for k in edges:
                        indices.append(self.all_edges.index(k))
                    for k in range(len(self.mind)):
                        self.mind[k].topology_changes(indices)
                self.prev_edges = edges
                if self.all_edges is None:
                    self.all_edges = edges.copy()


    def update_adj_matrix(self, adj_matrix):
        self.adj_matrix = adj_matrix

    def define_network(self):

        # init state
        self.link_state_initializer = LinkUpdate(self.num_features, self.first_hidden_layer_size, self.final_hidden_layer_size, self.link_state_size,
                                                 self.hidden_layer_initializer, self.hidden_layer_initializer_gain, self.activation_fn)

        # message
        self.create_message_2 = CreateMessage(self.message_input_shape, self.message_hidden_layer_size, self.link_state_size,
                                            self.hidden_layer_initializer, self.hidden_layer_initializer_gain,
                                            self.activation_fn)

        self.create_message_1 = CreateMessage(self.message_input_shape, self.message_hidden_layer_size, self.link_state_size,
                                            self.hidden_layer_initializer, self.hidden_layer_initializer_gain,
                                            self.activation_fn) if self.networks[0] == '1' else self.create_message_2

        self.create_message_3 = CreateMessage(self.message_input_shape, self.message_hidden_layer_size, self.link_state_size,
                                            self.hidden_layer_initializer, self.hidden_layer_initializer_gain,
                                            self.activation_fn) if self.networks[1] == '3' else self.create_message_1 if self.networks[1] == '1' else self.create_message_2

        # link update
        self.link_update_2 = LinkUpdate(self.link_input_shape_2, self.first_hidden_layer_size, self.final_hidden_layer_size, self.link_state_size,
                                        self.hidden_layer_initializer, self.hidden_layer_initializer_gain, self.activation_fn)

        self.link_update_1 = LinkUpdate(self.link_input_shape_1, self.first_hidden_layer_size, self.final_hidden_layer_size, self.link_state_size,
                                        self.hidden_layer_initializer, self.hidden_layer_initializer_gain, self.activation_fn) if self.networks[2] == '1' else self.link_update_2

        self.link_update_3 = LinkUpdate(self.link_input_shape_3, self.first_hidden_layer_size, self.final_hidden_layer_size, self.link_state_size,
                                        self.hidden_layer_initializer, self.hidden_layer_initializer_gain, self.activation_fn) if self.networks[3] == '3' else self.link_update_1 if self.networks[3] == '1' else self.link_update_2


        # readout
        self.readout = ReadOut(self.readout_input_shape, self.first_hidden_layer_size, self.final_hidden_layer_size, self.num_actions,
                               self.hidden_layer_initializer, self.hidden_layer_initializer_gain,
                               self.final_layer_initializer, self.final_layer_initializer_gain,
                               self.activation_fn, self.final_activation_fn, self.kernel_regularizer,
                               self.kernel_regularizer_rate, self.dropout_rate)

    def local_aggregation(self, link_states, first_iter=False): # aggregate local states
        # TODO
        if first_iter == True:
            link_states = link_states[:, :2]
        if self.memory_lookup == 'state': # in this mode, each agent decides whether or not to use memory independently of its neighbors
            return link_states
        outcoming_link_states_neigh = torch.index_select(input = link_states, dim = 0, index = self.outcoming_links_2)
        message_inputs_neigh = torch.cat([outcoming_link_states_neigh], dim = 1).to(torch.float32)
        aggregated_messages = self.message_aggregation(None, None, message_inputs_neigh, mode='n')
        aggregated_loc_states = torch.cat([link_states, aggregated_messages], dim=1).to(torch.float32)
        return aggregated_loc_states

    def get_memory_states(self, link_states, first_iter=False): # calculate states that are used to decide whether or not to use memory
        if self.mode == '1' or self.mode == '2' or self.mode == '3':
            memory_states = self.local_aggregation(link_states, first_iter=first_iter)
        else: # in modes 4 and 5, local neighbors are processed through a message exchange iteration
            incoming_link_states_neigh = torch.index_select(input = link_states, dim = 0, index = self.incoming_links_2)
            outcoming_link_states_neigh = torch.index_select(input = link_states, dim = 0, index = self.outcoming_links_2)
            message_inputs_neigh = torch.cat([incoming_link_states_neigh, outcoming_link_states_neigh], dim = 1).to(torch.float32)
            messages_neigh = self.create_message_1(message_inputs_neigh)
            aggregated_messages = self.message_aggregation(None, None, messages_neigh, mode='n')
            link_update_input = torch.cat([link_states, aggregated_messages], dim=1).to(torch.float32)
            memory_states = self.link_update_1(link_update_input)
        return memory_states

    def message_passing(self, input_tensor): # input_tensor - vector of length n_links*2

        # padding
        link_states = torch.reshape(input_tensor, (self.num_features, self.n_links))
        link_states = link_states.T
        if self.states_logging and not self.training and not self.eval_mode:
            self.log_states[0].append(link_states.detach().cpu().numpy())
        # padding = (0, self.link_state_size - self.num_features) # was in MATE before
        # link_states = torch.nn.functional.pad(link_states, padding) # was in MATE before
        link_states = self.link_state_initializer(link_states)
        # link_states - tensor of shape (n_links, 16)
        codes = []


        # message iterations
        for message_iteration in range(self.message_iterations):

            # Step 1 - internal message exchange, deciding whether or not to use memory
            if self.memory_used or self.mode == '4' or self.mode == '5':
                memory_states = self.get_memory_states(link_states, first_iter = message_iteration == 0) # link's states aggregated with local states
                # memory states - either 2, 6, 16, or 48 long vectors
            if self.memory_used and not self.training:
                code, policy_memory = self.mind[message_iteration].update_memory(memory_states.detach().cpu().numpy().astype(float), update=self.update_memory)
                codes.append(code)
                if code == ACT_VEC_USE or code == ACT_VEC_USE_AND_UPDATE:
                    if self.states_logging and not self.training and not self.eval_mode:
                        for i in range(message_iteration, self.message_iterations):
                            self.log_states[i + 1].append(link_states.detach().cpu().numpy())
                    return codes, torch.tensor(policy_memory, dtype=torch.float32, device=self.device)

            # Step 2 - External message passing
            # In modes 4 and 5, memory_states are 16-long vectors representing agent's local state,
            # and they are used for further message processing
            # In modes 1-3, memory_states are only used for memory lookup
            link_states_old = link_states # for external exchange, we don't need info about link's neighbors
            if self.mode == '4' or self.mode == '5':
                link_states = memory_states

            incoming_link_states = torch.index_select(input = link_states, dim = 0, index = self.incoming_links)
            outcoming_link_states = torch.index_select(input = link_states_old, dim = 0, index = self.outcoming_links)
            incoming_link_states_neigh = torch.index_select(input = link_states, dim = 0, index = self.incoming_links_2)
            outcoming_link_states_neigh = torch.index_select(input = link_states, dim = 0, index = self.outcoming_links_2)
            message_inputs_out = torch.cat([incoming_link_states, outcoming_link_states], dim = 1).to(torch.float32)
            # message_inputs_in = torch.cat([outcoming_link_states, incoming_link_states], dim = 1).to(torch.float32)
            message_inputs_neigh = torch.cat([incoming_link_states_neigh, outcoming_link_states_neigh], dim = 1).to(torch.float32)
            messages_out = self.create_message_2(message_inputs_out)
            # messages_in = self.create_message_2(message_inputs_in)
            messages_neigh = self.create_message_2(message_inputs_neigh)

            aggregated_messages = self.message_aggregation(messages_out, None, messages_neigh, mode = 'on' if self.mode == '1' or self.mode == '2' else 'o')
            link_update_input = torch.cat([link_states, aggregated_messages], dim=1).to(torch.float32)
            link_states = self.link_update_2(link_update_input)
            if self.states_logging and not self.training and not self.eval_mode:
                self.log_states[message_iteration + 1].append(link_states.detach().cpu().numpy())

            # Step 3 - another internal message passing
            if self.mode == '2' or self.mode == '3' or self.mode == '5':
                incoming_link_states_neigh = torch.index_select(input = link_states, dim = 0, index = self.incoming_links_2)
                outcoming_link_states_neigh = torch.index_select(input = link_states, dim = 0, index = self.outcoming_links_2)
                message_inputs_neigh = torch.cat([incoming_link_states_neigh, outcoming_link_states_neigh], dim = 1).to(torch.float32)
                messages_neigh = self.create_message_3(message_inputs_neigh)
                aggregated_messages = self.message_aggregation(None, None, messages_neigh, mode = 'n')
                link_update_input = torch.cat([link_states, aggregated_messages], dim=1).to(torch.float32)
                link_states = self.link_update_3(link_update_input)
        return codes, link_states

    def message_aggregation(self, messages_out, messages_in, messages_neigh, mode='on'):
        shape = [self.n_links] + list(messages_out.shape[1:] if messages_out is not None else messages_neigh.shape[1:])
        if self.aggregation == 'sum':

            out = torch.zeros(shape, dtype=messages_out.dtype, device=self.device)
            aggregated_messages = out.scatter_reduce(dim=0, index=self.outcoming_links.unsqueeze(-1).expand_as(messages_out), src=messages_out, reduce='sum')

        elif self.aggregation == 'min_max':
            if 'o' in mode or 'i' in mode:
                out1 = torch.full(shape, torch.iinfo(torch.int64).min, dtype=messages_out.dtype, device=self.device)
                # agg_max_in = out1.scatter_reduce(dim=0, index=torch.tensor(self.outcoming_links).unsqueeze(-1).expand_as(messages_in), src=messages_in, reduce='amax')
                agg_max_out = out1.scatter_reduce(dim=0, index=self.incoming_links.unsqueeze(-1).expand_as(messages_out), src=messages_out, reduce='amax')
                out2 = torch.full(shape, torch.iinfo(torch.int64).max, dtype=messages_out.dtype, device=self.device)
                # agg_min_in = out2.scatter_reduce(dim=0, index=torch.tensor(self.outcoming_links).unsqueeze(-1).expand_as(messages_in), src=messages_in, reduce='amin')
                agg_min_out = out2.scatter_reduce(dim=0, index=self.incoming_links.unsqueeze(-1).expand_as(messages_out), src=messages_out, reduce='amin')
            if 'n' in mode:
                out1_n = torch.full(shape, torch.iinfo(torch.int64).min, dtype=messages_neigh.dtype, device=self.device)
                out2_n = torch.full(shape, torch.iinfo(torch.int64).max, dtype=messages_neigh.dtype, device=self.device)
                agg_max_neigh = out1_n.scatter_reduce(dim=0, index=self.incoming_links_2.unsqueeze(-1).expand_as(messages_neigh), src=messages_neigh, reduce='amax')
                agg_min_neigh = out2_n.scatter_reduce(dim=0, index=self.incoming_links_2.unsqueeze(-1).expand_as(messages_neigh), src=messages_neigh, reduce='amin')
            if mode == 'n':
                aggregated_messages = torch.cat([agg_max_neigh, agg_min_neigh], dim=1)
                return aggregated_messages
            if mode == 'o':
                aggregated_messages = torch.cat([agg_max_out, agg_min_out], dim=1)
                return aggregated_messages
            # aggregated_messages = torch.cat([agg_max_in, agg_min_in, agg_max_out, agg_min_out, agg_max_neigh, agg_min_neigh], dim=1)
            # aggregated_messages = torch.cat([agg_max_out, agg_min_out], dim=1)
            aggregated_messages = torch.cat([agg_max_out, agg_min_out, agg_max_neigh, agg_min_neigh], dim=1)
        return aggregated_messages

    def forward(self, input_tensor):
        codes, link_states_t = self.message_passing(input_tensor)
        if self.memory_used and not self.training:
            self.memory_used = False
            states_logging = self.states_logging
            self.states_logging = False
            _, link_states_t_true = self.message_passing(input_tensor)
            self.states_logging = states_logging
            self.memory_used = True
        else:
            link_states_t_true = None
        # print("\ncodes", codes, end="")

        if self.memory_used and not self.training:
            link_states = link_states_t.detach().cpu().numpy()
            if codes[-1] == ACT_VEC_USE:
                policy = self.readout(link_states_t) # policy = link_states.astype('float32')
                codes = codes[:-1]
            else:
                policy = self.readout(link_states_t)
            self.log_message_iterations_done += len(codes) # Suppose, when code == ACT_VEC_USE, no message passing iteration is done
            self.log_message_iterations_possible += self.message_iterations
            if not self.eval_mode:
                self.log_message_iterations_done_agents.append(np.array([len(codes) for i in range(self.n_links)]))
            for idx, code in enumerate(codes):
                if code == ACT_VEC_UPDATE:
                    if self.update_memory == True: self.mind[idx].update_action(link_states) # self.mind[idx].update_action(policy)
                elif code == ACT_VEC_NEW_UPDATE:
                    if self.update_memory == True: self.mind[idx].update_action_new(link_states) # self.mind[idx].update_action_new(policy)
            policy_true = self.readout(link_states_t_true)
            policy_true = torch.reshape(policy_true, (-1,))
        else:
            policy = self.readout(link_states_t)
            policy_true = None
        # print("policy", policy)
        policy = torch.reshape(policy, (-1,))
        return policy, policy_true
