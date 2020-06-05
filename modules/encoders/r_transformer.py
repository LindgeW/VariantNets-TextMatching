import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import copy
import numpy as np
from ..layers import Dropout


def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


# class LayerNorm(nn.Module):
#     def __init__(self, features, eps=1e-6):
#         super(LayerNorm, self).__init__()
#         self.a_2 = nn.Parameter(torch.ones(features))
#         self.b_2 = nn.Parameter(torch.zeros(features))
#         self.eps = eps
#
#     def forward(self, x):
#         mean = x.mean(-1, keepdim=True)
#         std = x.std(-1, keepdim=True)
#         return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class SublayerConnection(nn.Module):
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = nn.LayerNorm(size)
        self.dropout = Dropout(dropout)

    def forward(self, x, sublayer):
        # return x + self.dropout(sublayer(self.norm(x)))
        return self.norm(x + self.dropout(sublayer(x)))


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        d_ff = d_model * 4
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))


def attention(query, key, value, mask=None, dropout=None):
    """
        Compute 'Scaled Dot Product Attention'
        query, key, value : batch_size, n_head, seq_len, dim of space
    """
    d_k = query.size(-1)
    # scores: batch_size, n_head, seq_len, seq_len
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)

    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)

    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn


class MHPooling(nn.Module):
    def __init__(self, d_model, h, dropout=0.1):
        super(MHPooling, self).__init__()
        assert d_model % h == 0
        # assuming d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = Dropout(p=dropout)

        # auto-regressive
        attn_shape = (1, 3000, 3000)
        subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
        self.register_buffer('mask', (torch.from_numpy(subsequent_mask) == 0).unsqueeze(1))

    def forward(self, x):
        nb_batch, seq_len, d_model = x.shape

        query, key, value = \
            [l(x).reshape(nb_batch, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (x, x, x))]

        x, self.attn = attention(query, key, value, mask=self.mask[:, :, :seq_len, :seq_len],
                                 dropout=self.dropout)

        x = x.transpose(1, 2).contiguous().reshape(nb_batch, -1, self.h * self.d_k)
        return self.linears[-1](x)


class LocalRNN(nn.Module):
    def __init__(self, input_dim, output_dim, rnn_type, ksize, dropout):
        super(LocalRNN, self).__init__()
        self.input_dim = input_dim
        self.ksize = ksize

        if rnn_type == 'GRU':
            self.rnn = nn.GRU(output_dim, output_dim, batch_first=True)
        elif rnn_type == 'LSTM':
            self.rnn = nn.LSTM(output_dim, output_dim, batch_first=True)
        else:
            self.rnn = nn.RNN(output_dim, output_dim, batch_first=True)

        self.output = nn.Sequential(nn.Linear(output_dim, output_dim), nn.ReLU())

        # To speed up
        idx = [i for j in range(self.ksize - 1, 10000, 1) for i in range(j - (self.ksize - 1), j + 1, 1)]
        self.register_buffer('select_index', torch.LongTensor(idx))
        self.register_buffer('zeros', torch.zeros(self.ksize - 1, input_dim))

    def get_K(self, x):
        batch_size, l, d_model = x.shape
        zeros = self.zeros.unsqueeze(0).repeat(batch_size, 1, 1)
        x = torch.cat((zeros, x), dim=1)
        key = torch.index_select(x, 1, self.select_index[:self.ksize * l])
        key = key.reshape(batch_size, l, self.ksize, -1)
        return key

    def forward(self, x):
        x = self.get_K(x)  # b x seq_len x ksize x d_model
        batch, l, ksize, d_model = x.shape
        h = self.rnn(x.reshape(-1, self.ksize, d_model))[0][:, -1, :]
        return h.reshape(batch, l, d_model)


class LocalRNNLayer(nn.Module):
    def __init__(self, input_dim, output_dim, rnn_type, ksize, dropout):
        super(LocalRNNLayer, self).__init__()
        self.local_rnn = LocalRNN(input_dim, output_dim, rnn_type, ksize, dropout)
        self.connection = SublayerConnection(output_dim, dropout)

    def forward(self, x):
        x = self.connection(x, self.local_rnn)
        return x


class Block(nn.Module):
    def __init__(self, input_dim, output_dim, rnn_type, ksize, N, h, dropout):
        super(Block, self).__init__()
        self.layers = clones(LocalRNNLayer(input_dim, output_dim, rnn_type, ksize, dropout), N)
        self.connections = clones(SublayerConnection(output_dim, dropout), 2)
        self.pooling = MHPooling(input_dim, h, dropout)
        self.feed_forward = PositionwiseFeedForward(input_dim, dropout)

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
        x = self.connections[0](x, self.pooling)
        x = self.connections[1](x, self.feed_forward)
        return x


class RTransformer(nn.Module):
    def __init__(self, d_model, rnn_type, ksize, n_level, rnn_level, nb_heads, dropout=0.1):
        super(RTransformer, self).__init__()
        self.d_model = d_model
        self.dropout = Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)
        # self.feed_forward = PositionwiseFeedForward(d_model, dropout)

        layers = []
        for i in range(n_level):
            layers.append(Block(d_model, d_model, rnn_type, ksize, N=rnn_level, h=nb_heads, dropout=dropout))

        self.forward_net = nn.Sequential(*layers)

    def forward(self, x, non_pad_mask=None):
        if non_pad_mask is not None:
            x = x * non_pad_mask.float().unsqueeze(-1)

        x = self.forward_net(x)

        return x
