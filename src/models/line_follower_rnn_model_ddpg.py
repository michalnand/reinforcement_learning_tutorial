import torch

class ModelActor(torch.nn.Module):

    def __init__(self, input_shape, outputs_count):
        super(ModelActor, self).__init__()

        self.gru        = torch.nn.GRU(input_shape[1], 64, batch_first = True)
        self.output     = torch.nn.Linear(64, outputs_count)
        self.act        = torch.nn.Tanh()
       
        torch.nn.init.uniform_(self.output.weight, -0.3, 0.3)
        torch.nn.init.zeros_(self.output.bias)
    
    #state shape = (batch, sequence, features)
    def forward(self, state):
        y, _ = self.gru(state)

        #take last seq output
        y = y[:,-1,:]

        y = self.output(y)
        y = self.act(y)

        return y


   


class ModelCritic(torch.nn.Module):
    def __init__(self, input_shape, outputs_count):
        super(ModelCritic, self).__init__()

        self.flatten= torch.nn.Flatten()

        self.gru    = torch.nn.GRU(input_shape[1]  + outputs_count, 128, batch_first = True)

        self.l0     = torch.nn.Linear(128, 64)
        self.act0   = torch.nn.ReLU()
        
        self.l1     = torch.nn.Linear(64, 1)

        torch.nn.init.xavier_uniform_(self.l0.weight)
        torch.nn.init.zeros_(self.l0.bias)

        #this init is important for DDPG !!!
        torch.nn.init.uniform_(self.l1.weight, -0.003, 0.003)
        torch.nn.init.zeros_(self.l1.bias)
        
    #state shape    = (batch, seq, features)
    #action shape   = (batch, action)
    def forward(self, state, action):
        
        #repeat action to shape (batch, seq, action)
        action_ = action.unsqueeze(1).repeat(1, state.shape[1], 1)

        #cat action with state        
        x = torch.cat([state, action_], dim=2)

        x, _ = self.gru(x)

        #take last seq output
        x = x[:,-1,:]

        x = self.l0(x)
        x = self.act0(x)

        x = self.l1(x)

        return x

   