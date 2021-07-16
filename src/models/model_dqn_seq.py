import torch

class ModelDQNSeq(torch.nn.Module):

    def __init__(self, input_shape, outputs_count):
        super(ModelDQNSeq, self).__init__()

        self.lstm     = torch.nn.LSTM(input_shape[1], 64, batch_first = True)
        self.output   = torch.nn.Linear(64, outputs_count)
       
        torch.nn.init.xavier_uniform_(self.output.weight)
        torch.nn.init.zeros_(self.output.bias)
        
    def forward(self, state):
        y, _ = self.lstm(state)

        #take last seq output
        y = y[:,-1,:]
        y = self.output(y)

        return y
