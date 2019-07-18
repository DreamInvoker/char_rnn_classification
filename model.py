import torch
from torch import nn


class RNN(nn.Module):

    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.i2o = nn.Linear(input_size + hidden_size, output_size)
        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input_data, hidden):
        concatenate = torch.cat((input_data, hidden), 1)
        hidden = self.i2h(concatenate)
        output = self.i2o(concatenate)
        output = self.softmax(output)
        return output, hidden

    def init_hidden(self):
        return torch.zeros((1, self.hidden_size))
