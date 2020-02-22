import torch
import torch.nn as nn

import numpy

class ModelDQN(torch.nn.Module):

    def __init__(self, input_shape, outputs_count, neurons_count = [64, 32]):
        super(ModelDQN, self).__init__()

        layer_inputs_count = input_shape[0]

        self.layers = []
        
        for i in range(len(neurons_count)):
            layer_outputs_count = neurons_count[i]

            self.layers.append(nn.Linear(layer_inputs_count, layer_outputs_count))
            self.layers.append(nn.ReLU())

            layer_inputs_count = layer_outputs_count

        self.layers.append(nn.Linear(layer_inputs_count, outputs_count))

        self.model = nn.Sequential(*self.layers)
        print(self.model)


    def forward(self, state):
        return self.model.forward(state)

    def get_q_values(self, state):
        with torch.no_grad():
            rs = numpy.reshape(state, (1, ) + state.shape)

            state_dev       = torch.tensor(rs, dtype=torch.float32).detach()
            network_output  = self.model.forward(state_dev)

            return network_output[0].detach().numpy()
