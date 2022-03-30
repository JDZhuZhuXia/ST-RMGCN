import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import sys
from embed import PositionalEmbedding
from RNN import RNN
from attention import *
import numpy as np
 

class nconv(nn.Module):
    def __init__(self):
        super(nconv,self).__init__()

    def forward(self,x, A):
        x = torch.einsum('ncvl,vw->ncwl',(x,A))
        return x.contiguous()

class linear(nn.Module):
    def __init__(self,c_in,c_out):
        super(linear,self).__init__()
        self.mlp = torch.nn.Conv2d(c_in, c_out, kernel_size=(1, 1), padding=(0,0), stride=(1,1), bias=True)

    def forward(self,x):
        return self.mlp(x)

class gcn(nn.Module):
    def __init__(self,c_in,c_out,dropout,support_len=3,order=2):
        super(gcn,self).__init__()
        self.c_out = c_out
        self.nconv = nconv()
        c_in = (order*support_len+1)*c_in
        self.mlp = linear(c_in,c_out)
        self.dropout = dropout
        self.order = order


    def forward(self,x,support):
        out = [x]
        for a in support:
            x1 = self.nconv(x,a)
            out.append(x1)
            for k in range(2, self.order + 1):
                x2 = self.nconv(x1,a)
                out.append(x2)
                x1 = x2
        
        h = torch.cat(out,dim=1)
        h = self.mlp(h)
        h = F.dropout(h, self.dropout, training=self.training)
        return h
class feedforward(nn.Module):
    def __init__(self,c_in,c_out):
        super(feedforward,self).__init__()
        self.conv1 = nn.Conv2d(c_in,c_out,kernel_size=(1,1),padding=(0,0), stride=(1,1), bias=True)
        self.conv2 = nn.Conv2d(c_out,c_out,kernel_size=(1,1), padding=(0,0), stride=(1,1),bias=True)
    def forward(self,x):
        x = self.conv1(x)
        x = self.conv2(F.gelu(x))
        return x
    
class timeAttentionLayer(nn.Module):
    def __init__(self,attention, d_model, d_ff=None, dropout=0.1, activation="gelu"):
        super(timeAttentionLayer, self).__init__()
        d_ff = d_ff or 4*d_model
        self.attention = attention
        self.conv1 = nn.Conv2d(in_channels=d_model, out_channels=d_ff, kernel_size=(1,1))
        self.conv2 = nn.Conv2d(in_channels=d_ff, out_channels=d_model, kernel_size=(1,1))
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu
    def forward(self, x, attn_mask=None):
        new_x, attn = self.attention(
            x, x, x,
            attn_mask = attn_mask
        )
        x = x + self.dropout(new_x)
        y = x = self.norm1(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1,1))))
        y = self.dropout(self.conv2(y).transpose(-1,1))
        return self.norm2(x+y), attn

class timeAttention(nn.Module):
    def __init__(self, attn_layers, conv_layers=None, norm_layer=None):
        super(timeAttention, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.conv_layers = nn.ModuleList(conv_layers) if conv_layers is not None else None
        self.norm = norm_layer
    def forward(self, x, attn_mask=None):
        # x [B, L, D]
        x = x.permute(0,2,3,1)
        attns = []
        if self.conv_layers is not None:
            for attn_layer, conv_layer in zip(self.attn_layers, self.conv_layers):
                x, attn = attn_layer(x, attn_mask=attn_mask)
                x = conv_layer(x)
                attns.append(attn)
            x, attn = self.attn_layers[-1](x)
            attns.append(attn)
        else:
            for attn_layer in self.attn_layers:
                x, attn = attn_layer(x, attn_mask=attn_mask)
                attns.append(attn)

        if self.norm is not None:
            x = self.norm(x)
        x = x.permute(0,3,1,2)
        return x, attns
class ConvLayer(nn.Module):
    def __init__(self, c_in):
        super(ConvLayer, self).__init__()
        self.downConv = nn.Conv2d(in_channels=c_in,
                                  out_channels=c_in,
                                  kernel_size=(3,1),
                                  padding=(1,0),
                                  padding_mode='zeros')
        self.norm = nn.BatchNorm2d(c_in)
        self.activation = nn.ELU()
        self.maxPool = nn.MaxPool2d(kernel_size=(3,1), stride=(2,1), padding=(1,0))

    def forward(self, x):
        x = self.downConv(x.permute(0, 3, 2, 1))
        x = self.norm(x)
        x = self.activation(x)
        x = self.maxPool(x)
        x = x.transpose(1,3)
        return x


class RMGCN(nn.Module):
    def __init__(self, device, num_nodes, dropout=0.3, supports=None, gcn_bool=True, addaptadj=True, adddymadj=True, aptinit=None, in_dim=2,out_dim=12,residual_channels=64,dilation_channels=64,skip_channels=256,end_channels=512,kernel_size=2,blocks=4,layers=2,dim=40,head=4):
        super(RMGCN, self).__init__()
        self.dropout = dropout
        self.blocks = blocks
        self.layers = layers
        self.head = head
        self.gcn_bool = gcn_bool
        self.addaptadj = addaptadj
        self.adddymadj = adddymadj

        self.filter_convs = nn.ModuleList()
        self.gate_convs = nn.ModuleList()
        self.residual_convs = nn.ModuleList()
        self.skip_convs = nn.ModuleList()
        self.bn = nn.ModuleList()
        self.gconv = nn.ModuleList()
        self.linear_skip = nn.ModuleList()
        self.GRU = nn.ModuleList()
        self.linear_b = nn.ModuleList()
        self.linear_p = nn.ModuleList()

        self.val_feature = nn.Conv2d(in_channels=1,
                                    out_channels=residual_channels//2,
                                    kernel_size=(1,1))
        self.time_feature = nn.Conv2d(in_channels=5,
                                    out_channels=residual_channels//2,
                                    kernel_size=(1,1))
        self.position_attn = PositionalEmbedding(residual_channels)
        self.start_conv = nn.Conv2d(in_channels=residual_channels,
                                    out_channels=residual_channels,
                                    kernel_size=(1,1))

        self.supports = supports

        e_layers = 1
        d_model =residual_channels 
        self.time_attention = timeAttention(
            [
                timeAttentionLayer(
                    AttentionLayer(FullAttention(True, attention_dropout=0.1, output_attention=True), 
                                d_model, n_heads = 4, d_keys=d_model, d_values=d_model),
                    d_model,
#                     d_ff,
#                     dropout=0.1,
#                     activation=activation
                ) for l in range(e_layers)
            ],
            [
                ConvLayer(
                    d_model
                ) for l in range(e_layers-1)
            ] ,
            norm_layer=torch.nn.LayerNorm(d_model)
        )
        self.supports_len = 0
        if supports is not None:
            self.supports_len += len(supports)

        if gcn_bool and addaptadj:
            if aptinit is None:
                if supports is None:
                    self.supports = []
                self.nodevec1 = nn.Parameter(torch.randn(num_nodes, 10).to(device), requires_grad=True).to(device)
                self.nodevec2 = nn.Parameter(torch.randn(10, num_nodes).to(device), requires_grad=True).to(device)
                self.supports_len +=1
            else:
                if supports is None:
                    self.supports = []
                m, p, n = torch.svd(aptinit)
                initemb1 = torch.mm(m[:, :10], torch.diag(p[:10] ** 0.5))
                initemb2 = torch.mm(torch.diag(p[:10] ** 0.5), n[:, :10].t())
                self.nodevec1 = nn.Parameter(initemb1, requires_grad=True).to(device)
                self.nodevec2 = nn.Parameter(initemb2, requires_grad=True).to(device)
                self.supports_len += 1
        if adddymadj:
            self.matrix_1 = nn.Parameter(torch.randn(24, dim).to(device), requires_grad=True).to(device)
            self.matrix_2 = nn.Parameter(torch.randn(num_nodes, dim).to(device), requires_grad=True).to(device)
            self.matrix_3 = nn.Parameter(torch.randn(num_nodes, dim).to(device), requires_grad=True).to(device)
            self.tensor_k = nn.Parameter(torch.randn(dim, dim, dim).to(device), requires_grad=True).to(device)
            self.supports_len +=1
        
        for i in range(self.head):
            self.GRU.append(RNN(num_nodes, residual_channels, residual_channels))
            self.linear_p.append(feedforward(residual_channels,residual_channels))
        self.bn_t = nn.BatchNorm2d(residual_channels)
        for b in range(blocks):
            for i in range(layers):
                    # 1x1 convolution for residual connection
                self.residual_convs.append(nn.Conv1d(in_channels=dilation_channels,
                                                         out_channels=residual_channels,
                                                         kernel_size=(1, 1)))

                    # 1x1 convolution for skip connection
                self.skip_convs.append(nn.Conv1d(in_channels=dilation_channels,
                                                     out_channels=skip_channels,
                                                     kernel_size=(1, 1)))
                self.bn.append(nn.BatchNorm2d(residual_channels))

                if self.gcn_bool:
                    self.gconv.append(gcn(dilation_channels,residual_channels,dropout,support_len=self.supports_len))

        self.end_conv_1 = nn.Conv2d(in_channels=skip_channels,
                                  out_channels=end_channels,
                                  kernel_size=(1,1),
                                  bias=True)

        self.end_conv_2 = nn.Conv2d(in_channels=end_channels,
                                    out_channels=out_dim,
                                    kernel_size=(1,1),
                                    bias=True)

    def dgconstruct(self, time_embedding, source_embedding, target_embedding, core_embedding):
        adp = torch.einsum('ai, ijk->ajk', time_embedding, core_embedding)
        adp = torch.einsum('bj, ajk->abk', source_embedding, adp)
        adp = torch.einsum('ck, abk->abc', target_embedding, adp)
        adp = F.softmax(F.relu(adp), dim=2)
        return adp

    def forward(self, input):
        in_len = input.size(3)
        x = input
        new_supports = None
        if self.gcn_bool and self.addaptadj and self.supports is not None:
            adp = F.softmax(F.relu(torch.mm(self.nodevec1, self.nodevec2)), dim=1)
            new_supports = self.supports + [adp]

        
        if self.adddymadj:
            ind = x[:,6,0,0].int().to('cpu').numpy().tolist()
            adp = self.dgconstruct(self.matrix_1[ind], self.matrix_2, self.matrix_3, self.tensor_k)
            adj_f = self.supports[0].unsqueeze(0).expand(x.shape[0],x.shape[2],x.shape[2])
            adj_b = self.supports[1].unsqueeze(0).expand(x.shape[0],x.shape[2],x.shape[2])
            new_supports = [adj_f, adj_b, adp]

#input Layer
        x_posi_attn = self.position_attn(x)
        global_feature = self.time_feature(x[:,1:6,:,:])
        x = torch.cat((self.val_feature(x[:,0:1,:,:]) ,global_feature),dim=1)
        x = self.start_conv(x)
        
        x = x + x_posi_attn
        x,attn = self.time_attention(x,attn_mask=None)

#muilt_head GCN
        skip_t = 0
        residual_t = x
        for i in range(self.head):
            init_state = self.GRU[i].init_hidden(x.shape[0])
            x, _ = self.GRU[i](residual_t.permute(0,3,2,1),init_state)
            x = x[:,-1,:,:].unsqueeze(1).permute(0,3,2,1)
            x_p = self.linear_p[i](x)
            skip_t = skip_t + x_p
        x = self.bn_t(skip_t)
        
        # 残差GCN
        skip = 0
        for i in range(self.blocks * self.layers):
            residual = x
            # parametrized skip connection
            s = x
            s = self.skip_convs[i](s)
            skip = s + skip
            if self.gcn_bool and self.supports is not None:
                if self.addaptadj or self.adddymadj:
                    x = self.gconv[i](residual, new_supports)
                else:
                    x = self.gconv[i](residual,self.supports)
            else:
                x = self.residual_convs[i](residual)

            x = x + residual
            x = self.bn[i](x)

        x = F.gelu(skip)
        x = F.gelu(self.end_conv_1(x))
        x = self.end_conv_2(x)
        return x

