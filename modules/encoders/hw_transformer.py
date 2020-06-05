import numpy as np
import torch
import torch.nn as nn
from ..scale_mix import ScalarMix


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
        self.dropout = nn.Dropout(dropout)

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
            att_weights = att_weights.masked_fill(att_mask, -1e9)  # float('-inf')

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
    def __init__(self, d_model, d_head, nb_heads, dropout=0.1, glu_type=1, use_bn=False):
        super(MultiHeadAttention, self).__init__()
        self.d_model = d_model
        self.d_head = d_head
        self.nb_heads = nb_heads
        self.glu_type = glu_type

        self.q_net = nn.Linear(in_features=d_model, out_features=d_head * nb_heads)
        self.kv_net = nn.Linear(in_features=d_model, out_features=2 * d_head * nb_heads)
        self.out_net = nn.Linear(in_features=d_head * nb_heads, out_features=d_model)
        self.att_layer = SelfAttention(d_head, dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

        self.use_bn = use_bn
        if use_bn:
            self.bn = nn.BatchNorm1d(d_model)
        self.act = nn.Sigmoid()  # nn.Tanh()
        self.fc1 = nn.Linear(d_model, d_model)
        self.fc2 = nn.Linear(d_model, d_model)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.q_net.weight, gain=1 / 2 ** 0.5)
        nn.init.xavier_uniform_(self.kv_net.weight, gain=1 / 2 ** 0.5)
        nn.init.xavier_uniform_(self.out_net.weight)

    def forward(self, h, att_mask=None):
        '''
        :param h: [bz, len, d_model]
        :param att_mask: [bz, len]
        :return: [bz, len, d_model]
        '''
        head_q = self.q_net(h)
        head_k, head_v = torch.chunk(self.kv_net(h), 2, -1)

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
        att_vec = att_out.transpose(1, 2).reshape(h.size(0), h.size(1), -1)
        # [bz, len_q, nb_heads*d_v] -> [bz, len_q, d_model]
        att_out = self.out_net(att_vec)
        if self.training:
            att_out = self.dropout(att_out)

        if self.glu_type == 1:
            g = self.act(self.fc1(h)) * self.fc2(h)
            if self.use_bn:
                g = self.bn(g.transpose(1, 2)).transpose(1, 2)

            output = self.layer_norm(h + att_out + g)
        elif self.glu_type == 2:
            Tx = self.act(self.fc1(h))
            output = self.layer_norm((1 - Tx) * h + Tx * self.fc2(h) + att_out)
        elif self.glu_type == 3:
            Tx = self.act(self.fc1(h))
            output = self.layer_norm(h + Tx * self.fc2(h) + (1 - Tx) * att_out)
        else:
            output = self.layer_norm(h + att_out)

        return output


class PositionwiseFF(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1, glu_type=1, use_bn=False):
        super(PositionwiseFF, self).__init__()

        self.glu_type = glu_type
        self.ffn = nn.Sequential(
            nn.Linear(in_features=d_model, out_features=d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(in_features=d_ff, out_features=d_model),
            nn.Dropout(dropout)
        )

        self.use_bn = use_bn
        if use_bn:
            self.bn = nn.BatchNorm1d(d_model)

        self.act = nn.Sigmoid()   # nn.Tanh()
        self.fc1 = nn.Linear(d_model, d_model)
        self.fc2 = nn.Linear(d_model, d_model)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, inputs):
        '''
        :param inputs: [bz, len_q, d_model]
        :return: [bz, len_q, d_model]
        '''
        if self.glu_type == 1:
            g = self.act(self.fc1(inputs)) * self.fc2(inputs)
            if self.use_bn:
                g = self.bn(g.transpose(1, 2)).transpose(1, 2)
            ff_out = self.ffn(inputs)
            output = self.layer_norm(inputs + ff_out + g)
        elif self.glu_type == 2:
            Tx = self.act(self.fc1(inputs))
            ff_out = self.ffn(inputs)
            output = self.layer_norm((1-Tx)*inputs + Tx*self.fc2(inputs) + ff_out)
        elif self.glu_type == 3:
            Tx = self.act(self.fc1(inputs))
            ff_out = self.ffn(inputs)
            output = self.layer_norm(inputs + Tx*self.fc2(inputs) + (1-Tx)*ff_out)
        else:
            ff_out = self.ffn(inputs)
            output = self.layer_norm(inputs + ff_out)

        return output


class EncoderLayer(nn.Module):
    def __init__(self, d_model, d_head, d_ff, nb_heads, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self._multi_head_att = MultiHeadAttention(d_model=d_model,
                                                  d_head=d_head,
                                                  nb_heads=nb_heads,
                                                  dropout=dropout)
        self._pwffn = PositionwiseFF(d_model=d_model,
                                      d_ff=d_ff,
                                      dropout=dropout)

    def forward(self, enc_in, att_mask=None, seq_mask=None):
        '''
        :param enc_in: [bz, len_k, d_model]
        :param att_mask: [bz, len_k] attention mask, 将attention填充部分mask
        :param seq_mask: [bz, len_q, 1] 将序列填充部分mask
        :return: [bz, len_q, d_model]
        '''
        # [bz, len_q, d_model]
        multi_head = self._multi_head_att(enc_in, att_mask)
        if seq_mask is not None:
            multi_head *= seq_mask.float()

        # [bz, len_q, d_model]
        out = self._pwffn(multi_head)
        if seq_mask is not None:
            out *= seq_mask.float()

        return out


class HWTransformer(nn.Module):
    def __init__(self, max_pos_embeddings, nb_layers, nb_heads, d_model, d_inner, dropout=0.1, att_drop=0.1):
        super(HWTransformer, self).__init__()

        d_head = d_model // nb_heads
        self.pos_embedding = nn.Embedding.from_pretrained(torch.from_numpy(PositionEncoding(max_pos_embeddings, d_model, pad_idx=0)), freeze=True)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

        self._encoder_stack = nn.ModuleList([
            EncoderLayer(d_model, d_head, d_inner, nb_heads, att_drop)
            for _ in range(nb_layers)
        ])

        self.dropout = nn.Dropout(dropout)
        self.scale = ScalarMix(nb_layers+1)

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
        if self.training:
            encoder_out = self.dropout(encoder_out)  # 0.1

        for encoder in self._encoder_stack:
            # [bz, len_q, d_model]
            encoder_out = encoder(encoder_out, att_mask=att_mask, seq_mask=seq_mask)
            # all_enc_outs.append(encoder_out)
        # encoder_out = self.scale(all_enc_outs)

        return encoder_out

