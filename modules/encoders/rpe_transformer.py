import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from ..scale_mix import ScalarMix
LAYER_NORM_EPS = 1e-6


class PositionalEmbedding(nn.Module):
    def __init__(self, d_model):
        super(PositionalEmbedding, self).__init__()
        inv_freq = 1 / (10000 ** (torch.arange(0.0, d_model, 2.0) / d_model))
        self.register_buffer('inv_freq', inv_freq)

    def forward(self, pos_seq):
        '''
        :param pos_seq: (seq_len, )
        :return:
        '''
        sinusoid_inp = torch.ger(pos_seq, self.inv_freq)
        pos_emb = torch.cat(tuple([sinusoid_inp.sin(), sinusoid_inp.cos()]), dim=-1)
        return pos_emb[:, None, :]  # (seq_len, 1, d_model)


class Dropout(nn.Module):
    def __init__(self, p: float = 0.0):
        super(Dropout, self).__init__()
        self.p = p
        self._drop = nn.Dropout(p)

    def forward(self, x):
        if self.training and self.p > 0:
            x = self._drop(x)
        return x


class GELU(nn.Module):
    def __init__(self):
        super(GELU, self).__init__()

    def forward(self, x):
        return 0.5 * x * (1. + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
        # return 0.5 * x * (1.0 + torch.erf(x / math.sqrt(2.0)))


class PositionwiseFF(nn.Module):
    def __init__(self, d_model, d_inner, dropout=0.0, pre_norm=False):
        super(PositionwiseFF, self).__init__()
        
        self.core = nn.Sequential(
            nn.Linear(d_model, d_inner),
            nn.ReLU(inplace=True),
            # GELU(),
            Dropout(dropout),
            nn.Linear(d_inner, d_model),
            Dropout(dropout)
        )

        self.layer_norm = nn.LayerNorm(d_model, eps=LAYER_NORM_EPS)

        self.pre_norm = pre_norm

    def forward(self, inp):
        if self.pre_norm:
            h = self.core(self.layer_norm(inp))
            output = inp + h
        else:
            h = self.core(inp)
            output = self.layer_norm(inp + h)
        return output


class RelMultiHeadAttn(nn.Module):
    def __init__(self, n_head, d_model, d_head, dropout=0.0, dropatt=0.0, pre_norm=False):
        super(RelMultiHeadAttn, self).__init__()

        self.d_model = d_model
        self.n_head = n_head
        self.d_head = d_head
        self.scale = 1. / (d_head**0.5)

        self.qkv_net = nn.Linear(d_model, 3*n_head*d_head, bias=False)
        self.r_net = nn.Linear(d_model, n_head*d_head, bias=False)
        self.o_net = nn.Linear(n_head*d_head, d_model, bias=False)

        self.dropatt = Dropout(dropatt)
        self.dropout = Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=LAYER_NORM_EPS)
        self.pre_norm = pre_norm

    def _rel_shift(self, x, zero_triu=False):
        (T, C), tail = x.shape[:2], x.shape[2:]
        zero_pad = torch.zeros((T, 1) + tail, device=x.device, dtype=x.dtype)
        x_padded = torch.cat(tuple([zero_pad, x]), dim=1)
        x_padded = x_padded.reshape((C+1, T) + tail)
        x = x_padded[1:].reshape_as(x)

        if zero_triu:
            ones = torch.ones((x.size(0), x.size(1)), device=x.device)
            x = x * torch.tril(ones, x.size(1) - x.size(0))[:, :, None, None]

        return x

    def forward(self, h, r, r_w_bias, r_r_bias, att_mask=None):
        qlen, rlen, bsz = h.size(0), r.size(0), h.size(1)
        n_head, d_head = self.n_head, self.d_head

        if self.pre_norm:
            w_heads = self.qkv_net(self.layer_norm(h))
        else:
            w_heads = self.qkv_net(h)

        w_head_q, w_head_k, w_head_v = torch.chunk(w_heads, 3, dim=-1)
        r_head_k = self.r_net(r)

        klen = w_head_k.size(0)
        w_head_q = w_head_q.reshape(qlen, bsz, n_head, d_head)
        w_head_k = w_head_k.reshape(klen, bsz, n_head, d_head)
        w_head_v = w_head_v.reshape(klen, bsz, n_head, d_head)
        r_head_k = r_head_k.reshape(rlen, n_head, d_head)

        rw_head_q = w_head_q + r_w_bias                                    # T x B x n_head x d_head
        AC = torch.einsum('ibnd,jbnd->ijbn', rw_head_q, w_head_k)             # T x C x B x n_head
        rr_head_q = w_head_q + r_r_bias
        BD = torch.einsum('ibnd,jnd->ijbn', rr_head_q, r_head_k)              # T x C x B x n_head
        BD = self._rel_shift(BD)
        # [qlen x klen x bsz x n_head]
        attn_score = (AC + BD).mul(self.scale)

        if att_mask is not None:  # 3-dim
            if att_mask.dim() == 2:
                attn_score = attn_score.float().masked_fill(
                        att_mask[None, :, :, None], -1e9).type_as(attn_score)

            elif att_mask.dim() == 3:
                attn_score = attn_score.float().masked_fill(
                    att_mask[:, :, :, None], -1e9).type_as(attn_score)

        # [qlen x klen x bsz x n_head]
        attn_prob = F.softmax(attn_score, dim=1)
        attn_prob = self.dropatt(attn_prob)
        attn_vec = torch.einsum('ijbn,jbnd->ibnd', attn_prob, w_head_v)

        # [qlen x bsz x n_head x d_head]
        attn_vec = attn_vec.reshape(qlen, bsz, n_head * d_head)
        # [qlen x bsz x d_model]
        attn_out = self.o_net(attn_vec)
        attn_out = self.dropout(attn_out)

        if self.pre_norm:
            output = h + attn_out
        else:
            output = self.layer_norm(h + attn_out)

        return output


class RelDecoder(nn.Module):
    def __init__(self, n_head, d_model, d_head, d_inner, drop=0.1, dropatt=0.0, pre_norm=False):
        super(RelDecoder, self).__init__()

        self.attn = RelMultiHeadAttn(n_head, d_model, d_head, dropout=drop, dropatt=dropatt, pre_norm=pre_norm)
        self.ff = PositionwiseFF(d_model, d_inner, dropout=drop, pre_norm=pre_norm)

    def forward(self, h, r, r_w_bias, r_r_bias, attn_mask=None):
        attn = self.attn(h, r, r_w_bias, r_r_bias, attn_mask)
        ff = self.ff(attn)
        return ff


class RPETransformer(nn.Module):
    def __init__(self, d_model, d_inner, nb_heads, nb_layers, dropout=0.1, batch_first=True):
        super(RPETransformer, self).__init__()
        d_head = d_model // nb_heads
        self.d_model = d_model
        self.drop = Dropout(dropout)
        self.layers = nn.ModuleList([RelDecoder(nb_heads, d_model, d_head, d_inner, pre_norm=False)
                                     for _ in range(nb_layers)])
        self.pos_emb = PositionalEmbedding(self.d_model)
        self.batch_first = batch_first

        self.r_r_bias = nn.Parameter(torch.zeros((nb_heads, d_head)))
        self.r_w_bias = nn.Parameter(torch.zeros((nb_heads, d_head)))
        # self.r_r_bias = nn.Parameter(torch.zeros((n_layer, n_head, d_head)))
        # self.r_w_bias = nn.Parameter(torch.zeros((n_layer, n_head, d_head)))
        nn.init.xavier_normal_(self.r_r_bias)
        nn.init.xavier_normal_(self.r_w_bias)
        self.layer_norm = nn.LayerNorm(d_model, eps=LAYER_NORM_EPS)
        self.scale = ScalarMix(nb_layers+1)

    def forward(self, h, seq_mask=None):
        """
        Args/returns:
            h: (T, B, d_model)
            seq_mask: (T, B): real value for 1 and padding for 0
        """
        if self.batch_first:
            h = h.transpose(0, 1)
            if seq_mask is not None:
                seq_mask = seq_mask.transpose(0, 1)

        T, B, d_model = h.size()
        assert d_model == self.d_model

        pad_mask = None if seq_mask is None else ~seq_mask[None]
        all_ones = h.new_ones((T, T)).byte()
        attn_mask = torch.triu(all_ones, 1)[:, :, None]
        attn_mask = (attn_mask + pad_mask) > 0

        pos_seq = torch.arange(T-1, -1, -1., device=h.device, dtype=h.dtype)
        pos_embed = self.pos_emb(pos_seq)
        h = self.layer_norm(h)
        # hids = [h.transpose(0, 1)]

        h_out = self.drop(h)
        pos_embed = self.drop(pos_embed)

        for i, layer in enumerate(self.layers):
            h_out = layer(h_out, pos_embed, self.r_w_bias, self.r_r_bias, attn_mask=attn_mask)
            # hids.append(h_out.transpose(0, 1))

        # h_out = self.scale(hids)
        h_out = self.drop(h_out)

        if self.batch_first:
            h_out = h_out.transpose(0, 1)

        return h_out



