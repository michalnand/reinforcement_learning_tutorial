import torch

class ModelActor(torch.nn.Module):

    def __init__(self, input_shape, outputs_count):
        super(ModelActor, self).__init__()  

        self.lstm   = torch.nn.LSTM(input_shape[1], 64, batch_first = True)
        self.l0     = torch.nn.Linear(64, outputs_count)
        self.act0   = torch.nn.Tanh()

        #this init is important for DDPG !!!
        torch.nn.init.uniform_(self.l0.weight, -0.3, 0.3)
        torch.nn.init.zeros_(self.l0.bias)

    def forward(self, state):
        y, _ = self.lstm(state)

        #take last seq output from lstm
        y = y[:,-1,:] 

        y = self.l0(y)
        y = self.act0(y)
 
        return y

   


class ModelCritic(torch.nn.Module):
    def __init__(self, input_shape, outputs_count):
        super(ModelCritic, self).__init__()

        self.lstm   = torch.nn.LSTM(input_shape[1], 64, batch_first = True)

        self.l0     = torch.nn.Linear(64 + outputs_count, 64)
        self.act0   = torch.nn.ReLU()
        
        self.l1     = torch.nn.Linear(64, 1)

        torch.nn.init.xavier_uniform_(self.l0.weight)
        torch.nn.init.zeros_(self.l0.bias)

        #this init is important for DDPG !!!
        torch.nn.init.uniform_(self.l1.weight, -0.003, 0.003)
        torch.nn.init.zeros_(self.l1.bias)
        

    def forward(self, state, action):
        y, _ = self.lstm(state)

        y = y[:,-1,:]
        x = torch.cat([y, action], dim=1)

        x = self.l0(x)
        x = self.act0(x)

        x = self.l1(x)

        return x

   