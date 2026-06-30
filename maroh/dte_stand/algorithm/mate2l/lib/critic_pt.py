import numpy as np
import torch

def get_activation_method(activation_str):
    if activation_str == 'relu':
        return torch.nn.ReLU()
    elif activation_str == 'sigmoid':
        return torch.nn.Sigmoid()
    elif activation_str == 'tanh':
        return torch.nn.Tanh()
    elif activation_str == 'linear':
        return torch.nn.Identity()
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


class Critic(torch.nn.Module):
    def __init__(self,
                 graph,
                 num_features=2,
                 link_state_size=16,
                 aggregation='min_max',
                 first_hidden_layer_size=128,
                 dropout_rate=0.1,
                 final_hidden_layer_size=64,
                 message_iterations=8,
                 activation_fn='tanh',
                 final_activation_fn='linear',
                 device=torch.device('cpu')):
        super(Critic, self).__init__()
        # HYPERPARAMETERS
        self.num_features = num_features

        self.device = device

        self.link_state_size = link_state_size
        self.message_hidden_layer_size = final_hidden_layer_size
        self.aggregation = aggregation
        self.message_iterations = message_iterations
        self.message_input_shape = 2 * self.link_state_size
        if self.aggregation == 'sum':
            self.link_input_shape = 2 * self.link_state_size
        elif self.aggregation == 'min_max':
            self.link_input_shape = 3 * self.link_state_size

        self.num_readout_input_aggregations = 4
        self.readout_input_shape = self.link_state_size * self.num_readout_input_aggregations

        # FIXED INPUTS
        if graph:
            self.n_links = graph.number_of_edges()
            self.incoming_links = torch.tensor(graph.nodes()['graph_data']['incoming_links'], device=self.device)
            self.outcoming_links = torch.tensor(graph.nodes()['graph_data']['outcoming_links'], device=self.device)
            self.incoming_links_2 = torch.tensor(graph.nodes()['graph_data']['il2'], device=self.device)
            self.outcoming_links_2 = torch.tensor(graph.nodes()['graph_data']['ol2'], device=self.device)
            self.opposite_links = torch.tensor(graph.nodes()['graph_data']['opposites'], device=self.device)

        # NEURAL NETWORKS
        self.hidden_layer_initializer = torch.nn.init.orthogonal_
        self.hidden_layer_initializer_gain = np.sqrt(2)
        self.final_layer_initializer = torch.nn.init.orthogonal_
        self.final_layer_initializer_gain = 1
        self.kernel_regularizer = None
        self.kernel_regularizer_rate = 0.0
        self.activation_fn = activation_fn
        self.final_hidden_layer_size = final_hidden_layer_size
        self.first_hidden_layer_size = first_hidden_layer_size
        self.dropout_rate = dropout_rate
        self.final_activation_fn = final_activation_fn
        self.define_network()

    def update_message_iterations(self, message_iterations):
        self.message_iterations = message_iterations

    def update_graph(self, graph):
        self.n_links = graph.number_of_edges()
        self.incoming_links = torch.tensor(graph.nodes()['graph_data']['incoming_links'], device=self.device)
        self.outcoming_links = torch.tensor(graph.nodes()['graph_data']['outcoming_links'], device=self.device)
        self.incoming_links_2 = torch.tensor(graph.nodes()['graph_data']['il2'], device=self.device)
        self.outcoming_links_2 = torch.tensor(graph.nodes()['graph_data']['ol2'], device=self.device)
        self.opposite_links = torch.tensor(graph.nodes()['graph_data']['opposites'], device=self.device)

    def define_network(self):
        # message
        self.create_message = CreateMessage(self.message_input_shape, self.message_hidden_layer_size, self.link_state_size,
                                            self.hidden_layer_initializer, self.hidden_layer_initializer_gain,
                                            self.activation_fn)


        # link update
        self.link_update = LinkUpdate(self.link_input_shape, self.first_hidden_layer_size, self.final_hidden_layer_size, self.link_state_size,
                                      self.hidden_layer_initializer, self.hidden_layer_initializer_gain, self.activation_fn)


        # readout
        self.readout = ReadOut(self.readout_input_shape, self.first_hidden_layer_size, self.final_hidden_layer_size, 1,
                               self.hidden_layer_initializer, self.hidden_layer_initializer_gain,
                               self.final_layer_initializer, self.final_layer_initializer_gain,
                               self.activation_fn, self.final_activation_fn, self.kernel_regularizer,
                               self.kernel_regularizer_rate, self.dropout_rate)


    def message_passing(self, input_tensor):
        link_states = torch.reshape(input_tensor, (self.num_features, self.n_links))
        link_states = link_states.T
        padding = (0, self.link_state_size - self.num_features)
        link_states = torch.nn.functional.pad(link_states, padding)

        # message passing
        for _ in range(1):  # 4 from pseudocode # was range(self.message_iterations) in MATE before
            incoming_link_states = torch.index_select(input = link_states, dim = 0, index = self.incoming_links)
            outcoming_link_states = torch.index_select(input = link_states, dim = 0, index = self.outcoming_links)
            incoming_link_states_neigh = torch.index_select(input = link_states, dim = 0, index = self.incoming_links_2)
            outcoming_link_states_neigh = torch.index_select(input = link_states, dim = 0, index = self.outcoming_links_2)
            message_inputs = torch.cat([incoming_link_states, outcoming_link_states], dim = 1).to(torch.float32)
            messages = self.create_message(message_inputs)
            aggregated_messages = self.message_aggregation(messages)
            link_update_input = torch.cat([link_states, aggregated_messages], dim=1).to(torch.float32)
            link_states = self.link_update(link_update_input)
        return link_states

    def message_aggregation(self, messages, messages_neigh=None):
        shape = [self.n_links] + list(messages.shape[1:])
        if self.aggregation == 'sum':
            out = torch.zeros(shape, dtype=messages.dtype, device=self.device)
            aggregated_messages = out.scatter_reduce(dim=0, index=self.outcoming_links.unsqueeze(-1).expand_as(messages), src=messages, reduce='sum')

        elif self.aggregation == 'min_max':
            out1 = torch.full(shape, torch.iinfo(torch.int64).min, dtype=messages.dtype, device=self.device)
            agg_max = out1.scatter_reduce(dim=0, index=self.incoming_links.unsqueeze(-1).expand_as(messages), src=messages, reduce='amax')
            out2 = torch.full(shape, torch.iinfo(torch.int64).max, dtype=messages.dtype, device=self.device)
            agg_min = out2.scatter_reduce(dim=0, index=self.incoming_links.unsqueeze(-1).expand_as(messages), src=messages, reduce='amin')
            aggregated_messages = torch.cat([agg_max, agg_min], dim=1)
        return aggregated_messages

    def generate_readout_input(self, link_states):
        ls_mean = torch.mean(link_states, dim=0)
        ls_max = torch.max(link_states, dim=0).values
        ls_min = torch.min(link_states, dim=0).values
        ls_std = torch.std(link_states, dim=0, unbiased=False) # to make unbiased put false
        readout_input = torch.cat([ls_mean, ls_max, ls_min, ls_std], dim=0)
        readout_input = readout_input.unsqueeze(0)
        return readout_input

    def forward(self, input_tensor):
        link_states = self.message_passing(input_tensor)
        # link_states = torch.reshape(input_tensor, (self.num_features, self.n_links)) # was uncommented in MATE before
        # link_states = link_states.T # was uncommented in MATE before
        readout_input = self.generate_readout_input(link_states)
        V = self.readout(readout_input)
        V = V.reshape(-1)
        return V
