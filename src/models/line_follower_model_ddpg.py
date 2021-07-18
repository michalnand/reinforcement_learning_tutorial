import torch

class ModelActor(torch.nn.Module):

    def __init__(self, input_shape, outputs_count):
        super(ModelActor, self).__init__()

        self.flatten= torch.nn.Flatten()

        self.l0     = torch.nn.Linear(input_shape[0], 64)
        self.act0   = torch.nn.ReLU()
        
        self.l1     = torch.nn.Linear(64, 64)
        self.act1   = torch.nn.ReLU()
        
        self.l2     = torch.nn.Linear(64, outputs_count)
        self.act2   = torch.nn.Tanh()

        torch.nn.init.xavier_uniform_(self.l0.weight)
        torch.nn.init.zeros_(self.l0.bias)

        torch.nn.init.xavier_uniform_(self.l1.weight)
        torch.nn.init.zeros_(self.l1.bias)

        #this init is important for DDPG !!!
        torch.nn.init.uniform_(self.l2.weight, -0.3, 0.3)
        torch.nn.init.zeros_(self.l2.bias)
        

    def forward(self, state):
        s = self.flatten(state)

        x = self.l0(s)
        x = self.act0(x)

        x = self.l1(x)
        x = self.act1(x)

        x = self.l2(x)
        x = self.act2(x)

        return x

   


class ModelCritic(torch.nn.Module):
    def __init__(self, input_shape, outputs_count):
        super(ModelCritic, self).__init__()

        self.flatten= torch.nn.Flatten()

        self.l0     = torch.nn.Linear(input_shape[0] + outputs_count, 128)
        self.act0   = torch.nn.ReLU()
        
        self.l1     = torch.nn.Linear(128, 64)
        self.act1   = torch.nn.ReLU()
        
        self.l2     = torch.nn.Linear(64, 1)

        torch.nn.init.xavier_uniform_(self.l0.weight)
        torch.nn.init.zeros_(self.l0.bias)

        torch.nn.init.xavier_uniform_(self.l1.weight)
        torch.nn.init.zeros_(self.l1.bias)

        #this init is important for DDPG !!!
        torch.nn.init.uniform_(self.l2.weight, -0.003, 0.003)
        torch.nn.init.zeros_(self.l2.bias)
        

    def forward(self, state, action):
        s = self.flatten(state)

        x = torch.cat([s, action], dim=1)

        x = self.l0(x)
        x = self.act0(x)

        x = self.l1(x)
        x = self.act1(x)

        x = self.l2(x)

        return x

   