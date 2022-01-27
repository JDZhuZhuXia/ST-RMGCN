import torch
import torch.nn as nn


class RNNCell(nn.Module):
    def __init__(self, node_num, dim_in, dim_out):
        super(RNNCell, self).__init__()
        self.node_num = node_num
        self.hidden_dim = dim_out
        self.gate = nn.Linear(dim_in+self.hidden_dim, 2*dim_out, bias=True)
        self.update = nn.Linear(dim_in+self.hidden_dim, dim_out, bias=True)

    def forward(self, x, state):
        #x: B, num_nodes, input_dim
        #state: B, num_nodes, hidden_dim
        state = state.to(x.device)
        input_and_state = torch.cat((x, state), dim=-1)
        z_r = torch.sigmoid(self.gate(input_and_state))
        z, r = torch.split(z_r, self.hidden_dim, dim=-1)
        candidate = torch.cat((x, z*state), dim=-1)
        hc = torch.tanh(self.update(candidate))
        h = r*state + (1-r)*hc
        return h

    def init_hidden_state(self, batch_size):
        return torch.zeros(batch_size, self.node_num, self.hidden_dim)