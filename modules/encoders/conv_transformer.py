import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from .cnn_encoder import LightweightConv1d
from ..scale_mix import ScalarMix
from ..layers import Dropout


# PE(pos, 2i) = sin(pos/10000^(2i/d_model))
# PE(pos, 2i+1) = cos(pos/10000^(2i/d_model))
def PositionEncoding(max_len, d_model, pad_idx=None):
    pe = np.asarray([[pos / np.power(10000, 2.*(i//2) / d_model) for i in range(d_model)]
                     for pos in range(max_len)], dtype=np.float32)
    pe[:, 0::2] = np.sin(pe[:, 0::2])  # start : end : step
    pe[:, 1::2] = np.cos(pe[:, 1::2])
    if pad_idx is not None:
        pe[pad_idx] = 0
    return pe


class SelfAttention(nn.Module):
    def __init__(self, d_k, dropout=0.1):
        super(SelfAttention, self).__init__()
        self.softmax = nn.Softmax(dim=-1)
        self.scale = 1. / (d_k ** 0.5)
        self.dropout = Dropout(dropout)

    def forward(self, q, k, v, att_mask=None):
        '''
        :param q: [bz, len_q, Q]
        :param k: [bz, len_k, K]
        :param v: [bz, len_v, V]
        :param att_mask: [bz, len_q, len_k]  填充部分的mask
        more: Q==K, len_k==len_v
        :return: [bz, len_q, V]
        '''
        # [bz, len_q, Q] x [bz, K, len_k] -> [bz, len_q, len_k]
        att_weights = torch.matmul(q, k.transpose(-1, -2)).mul(self.scale)

        if att_mask is not None:
            att_weights = att_weights.masked_fill(att_mask.byte(), -1e9)  # float('-inf')

        # x = att_weights.new_full(att_weights.shape, -1e9, requires_grad=False)
        # fw_mask = x.tril(diagonal=-1)  # 下三角矩阵
        # bw_mask = x.triu(diagonal=1)  # 上三角矩阵
        # fw_att_weights = att_weights + fw_mask
        # bw_att_weights = att_weights + bw_mask

        # [bz, len_q, len_k]
        soft_att_weights = self.softmax(att_weights)
        # soft_fw_att_weights = self.softmax(fw_att_weights)
        # soft_bw_att_weights = self.softmax(bw_att_weights)

        if self.training:
            soft_att_weights = self.dropout(soft_att_weights)
            # soft_fw_att_weights = self.dropout(soft_fw_att_weights)
            # soft_bw_att_weights = self.dropout(soft_bw_att_weights)

        # [bz, len_q, len_k] * [bz, len_v, V] -> [bz, len_q, V]
        att_out = torch.matmul(soft_att_weights, v)
        # att_out = torch.matmul(soft_fw_att_weights, v) + torch.matmul(soft_bw_att_weights, v)
        # att_out = torch.cat((torch.matmul(soft_fw_att_weights, v) + torch.matmul(soft_bw_att_weights, v)), dim=-1)

        return att_out


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, d_head, nb_heads, bias=True, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        self.d_model = d_model
        self.d_head = d_head
        self.nb_heads = nb_heads

        self.q_net = nn.Linear(in_features=d_model, out_features=d_head * nb_heads, bias=bias)
        self.k_net = nn.Linear(in_features=d_model, out_features=d_head * nb_heads, bias=bias)
        self.v_net = nn.Linear(in_features=d_model, out_features=d_head * nb_heads, bias=bias)

        self.out_net = nn.Linear(in_features=d_head * nb_heads, out_features=d_model)
        self.att_layer = SelfAttention(d_head, dropout)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.q_net.weight, gain=1 / 2 ** 0.5)
        nn.init.xavier_uniform_(self.k_net.weight, gain=1 / 2 ** 0.5)
        nn.init.xavier_uniform_(self.v_net.weight, gain=1 / 2 ** 0.5)
        nn.init.xavier_uniform_(self.out_net.weight)

    def forward(self, h, att_mask=None):
        '''
        :param h: [bz, len, d_model]
        :param att_mask: [bz, len]
        :return: [bz, len, d_model]
        '''
        head_q = self.q_net(h)
        head_k = self.k_net(h)
        head_v = self.v_net(h)

        # [bz, len_q, d_k * nb_heads] -> [bz, nb_heads, len_q, d_k]
        q_fc = head_q.reshape(h.size(0), h.size(1), self.nb_heads, -1).transpose(1, 2)
        # [bz, len_k, d_k * nb_heads] -> [bz, nb_heads, len_k, d_k]
        k_fc = head_k.reshape(h.size(0), h.size(1), self.nb_heads, -1).transpose(1, 2)
        # [bz, len_v, d_v * nb_heads] -> [bz, nb_heads, len_v, d_v]
        v_fc = head_v.reshape(h.size(0), h.size(1), self.nb_heads, -1).transpose(1, 2)

        if att_mask is not None:
            # (bz, 1, 1, len_k)
            att_mask = att_mask[:, None, None, :]

        # (bz, nb_heads, len_q, d_v)
        att_out = self.att_layer(q_fc, k_fc, v_fc, att_mask)
        # (bz, len_q, nb_heads, d_v)
        att_out = att_out.transpose(1, 2).contiguous()
        att_out = att_out.reshape(att_out.size(0), att_out.size(1), -1)
        # [bz, len_q, nb_heads*d_v] -> [bz, len_q, d_model]
        att_out = self.out_net(att_out)

        return att_out


class PositionwiseFF(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFF, self).__init__()

        self.ffn = nn.Sequential(
            nn.Linear(in_features=d_model, out_features=d_ff),
            nn.ReLU(),
            Dropout(dropout),
            nn.Linear(in_features=d_ff, out_features=d_model),
        )

    def forward(self, inputs):
        '''
        :param inputs: [bz, len_q, d_model]
        :return: [bz, len_q, d_model]
        '''
        ff_out = self.ffn(inputs)

        return ff_out


class EncoderLayer(nn.Module):
    def __init__(self, args):
        super(EncoderLayer, self).__init__()

        self.embed_dim = args.embed_dim
        self.conv_dim = args.conv_dim
        self.multi_att_dim = args.d_model
        assert self.embed_dim == (self.conv_dim + self.multi_att_dim)

        self.dropout = args.dropout

        padding = args.kernel_size // 2 if args.kernel_size % 2 == 1 else ((args.kernel_size - 1) // 2, args.kernel_size // 2)
        self.conv_layer = LightweightConv1d(input_size=self.conv_dim,
                                            kernel_size=args.kernel_size,
                                            padding=padding)

        self.att_layer = MultiHeadAttention(d_model=self.multi_att_dim,
                                              d_head=args.d_model // args.nb_heads,
                                              nb_heads=args.nb_heads,
                                              dropout=args.att_drop)

        self.pwffn = PositionwiseFF(d_model=self.embed_dim,
                                      d_ff=args.d_ff,
                                      dropout=args.att_drop)

        self.layer_norm1 = nn.LayerNorm(self.embed_dim, eps=1e-6)
        self.layer_norm2 = nn.LayerNorm(self.embed_dim, eps=1e-6)

    def forward(self, x, att_mask=None, seq_mask=None):
        '''
        :param x: [bz, len_k, conv_dim + multi_att_dim]
        :param att_mask: [bz, len_k] attention mask, 将attention填充部分mask
        :param seq_mask: [bz, len_q, 1] 将序列填充部分mask
        :return: [bz, len_q, d_model]
        '''
        residual1 = x
        conv_in = x[..., :self.conv_dim]
        att_in = x[..., self.conv_dim: self.conv_dim+self.multi_att_dim].contiguous()

        conv_out = self.conv_layer(conv_in)
        att_out = self.att_layer(att_in, att_mask)
        x = torch.cat((conv_out, att_out), dim=-1)

        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.layer_norm1(x + residual1)

        if seq_mask is not None:
            x *= seq_mask.float()

        residual2 = x
        x = self.pwffn(x)

        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.layer_norm2(x + residual2)

        if seq_mask is not None:
            x *= seq_mask.float()

        return x


class ConvTransformer(nn.Module):
    def __init__(self, args):
        super(ConvTransformer, self).__init__()

        self.pos_embedding = nn.Embedding.from_pretrained(torch.from_numpy(PositionEncoding(args.max_pos_embeddings, args.embed_dim, pad_idx=0)), freeze=True)
        self.layer_norm = nn.LayerNorm(args.embed_dim, eps=1e-6)

        nb_layer = args.encoder_layer
        self._encoder_stack = nn.ModuleList([
            EncoderLayer(args) for _ in range(nb_layer)
        ])

        self.dropout = args.embed_drop
        self.scale = ScalarMix(nb_layer+1)

    def forward(self, embed_input, non_pad_mask=None):
        '''
        :param embed_input:  (bz, seq_len, embed_dim)
        :param non_pad_mask: (bz, seq_len)  pad部分为0
        :return:
        '''
        if non_pad_mask is None:
            att_mask = None
            seq_mask = None
        else:
            att_mask = (non_pad_mask == 0)   # 填充部分的mask(uint8类型)
            seq_mask = non_pad_mask[:, :, None]  # (bz, seq_len, 1)

        seq_range = torch.arange(embed_input.size(1), dtype=torch.long, device=embed_input.device) \
            .unsqueeze(dim=0)  # (1, seq_len)
        # [bz, seq_len, d_model]
        embed_input = embed_input + self.pos_embedding(seq_range)

        encoder_out = self.layer_norm(embed_input)
        # all_enc_outs = [encoder_out]
        encoder_out = F.dropout(encoder_out, p=self.dropout, training=self.training)

        for encoder in self._encoder_stack:
            # [bz, len_q, d_model]
            encoder_out = encoder(encoder_out, att_mask=att_mask, seq_mask=seq_mask)
            # all_enc_outs.append(encoder_out)
        # encoder_out = self.scale(all_enc_outs)

        return encoder_out

