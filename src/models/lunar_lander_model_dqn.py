import torch

class ModelDQN(torch.nn.Module):

    def __init__(self, input_shape, outputs_count):
        super(ModelDQN, self).__init__()

        self.l0     = torch.nn.Linear(input_shape[0], 64)
        self.act0   = torch.nn.ReLU()
        
        self.l1     = torch.nn.Linear(64, 64)
        self.act1   = torch.nn.ReLU()
        
        self.l2     = torch.nn.Linear(64, outputs_count)

        torch.nn.init.xavier_uniform_(self.l0.weight)
        torch.nn.init.zeros_(self.l0.bias)

        torch.nn.init.xavier_uniform_(self.l1.weight)
        torch.nn.init.zeros_(self.l1.bias)

        torch.nn.init.xavier_uniform_(self.l2.weight)
        torch.nn.init.zeros_(self.l2.bias)
        

    def forward(self, state):
        x = self.l0(state)
        x = self.act0(x)

        x = self.l1(x)
        x = self.act1(x)

        x = self.l2(x)

        return x

   