import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np


class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=15):
        super(PositionalEmbedding, self).__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float().to('cuda:0')

        pe.require_grad = False
        self.d_model = d_model
        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):

        out = self.pe[:, :x.shape[3]]
        out = torch.unsqueeze(out,dim=2)
        out_ = out.expand(1,out.size(1),x.shape[2],self.d_model)
        out_ = out_.expand(x.shape[0],out.size(1),x.shape[2],self.d_model)
        return out_.permute(0,3,2,1)


class TimeFeatureEmbedding(nn.Module):
    def __init__(self, d_model, freq='t'):
        super(TimeFeatureEmbedding, self).__init__()

        freq_map = {'h': 4, 't': 5, 's': 6, 'm': 1, 'a': 1, 'w': 2, 'd': 3, 'b': 3}
        d_inp = freq_map[freq]
        self.embed = nn.Linear(d_inp, d_model)

    def forward(self, x):
        out = self.embed(x)
        out = torch.reshape(out,(1,out.size(0),1,out.size(1)))
        out_ = out.expand(1,out.size(1),207,out.size(3))
        return out_


class TokenEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(TokenEmbedding, self).__init__()
        self.tokenConv = nn.Conv1d(in_channels=c_in,
                                   out_channels=d_model,
                                   kernel_size=(1,1) )

    def forward(self, x):
        x = self.tokenConv(x.permute(0,3,1,2)).permute(0,2,3,1)
        return x


class DataEmbedding(nn.Module):
    def __init__(self, c_in, d_model, dropout=0.1):
        super(DataEmbedding, self).__init__()

        # self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        # self.temporal_embedding = TimeFeatureEmbedding(d_model=d_model, freq=freq)

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        
        x_pos = self.position_embedding(x)

        out = np.concatenate((x,x_mark,x_pos),axis = 3)
        return out
