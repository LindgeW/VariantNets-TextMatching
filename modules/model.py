import torch
import torch.nn as nn
import torch.nn.functional as F
from .CPUEmbedding import CPUEmbedding
from .encoders.lstm import LSTM
from .encoders.tcn import TemporalConvNet
from .encoders.transformer import TransformerEncoder
from .encoders.rpe_transformer import RPETransformer
from .encoders.r_transformer import RTransformer
from .encoders.hw_transformer import HWTransformer
from .encoders.conv_transformer import ConvTransformer
from .layers import Dropout
'''
借助文本匹配任务来测试各类编码器
'''


class SentMatcher(nn.Module):
    def __init__(self, args, pre_embed):
        super(SentMatcher, self).__init__()
        self.embed_dim = args.embed_dim
        
        self.word_embed = nn.Embedding(num_embeddings=args.vocab_size,
                                       embedding_dim=self.embed_dim,
                                       padding_idx=0)
        self.pre_word_embed = CPUEmbedding(num_embeddings=pre_embed.shape[0],
                                            embedding_dim=self.embed_dim,
                                            padding_idx=0)
        self.pre_word_embed.weight.data.copy_(torch.from_numpy(pre_embed))
        self.pre_word_embed.weight.requires_grad = False
        # self.pre_word_embed = nn.Embedding.from_pretrained(torch.from_numpy(pre_embed), freeze=True)

        self.embed_drop = Dropout(args.embed_drop)
        # self.bn_embed = nn.BatchNorm1d(self.embed_dim)

        args.enc_type = 4
        if args.enc_type == 0:
            self.sent_encoder = LSTM(input_size=self.embed_dim, hidden_size=args.hidden_size, bidirectional=True)
            dec_in = 8 * args.hidden_size
        elif args.enc_type == 1:
            self.sent_encoder = TemporalConvNet(input_size=self.embed_dim,
                                                num_channels=args.num_channels,
                                                kernel_size=args.kernel_size)
            dec_in = 4 * args.num_channels[-1]
        elif args.enc_type == 2:
            self.sent_encoder = TransformerEncoder(max_pos_embeddings=args.max_pos_embeddings,
                                                   nb_layers=args.encoder_layer,
                                                   nb_heads=args.nb_heads,
                                                   d_model=self.embed_dim,
                                                   d_inner=args.d_ff)
            dec_in = 4 * self.embed_dim
        elif args.enc_type == 3:
            self.sent_encoder = RPETransformer(d_model=self.embed_dim,
                                               d_inner=args.d_ff,
                                               nb_heads=args.nb_heads,
                                               nb_layers=args.encoder_layer)
            dec_in = 4 * self.embed_dim
        elif args.enc_type == 4:
            # 参数设置与Transformer一致
            self.sent_encoder = RTransformer(d_model=self.embed_dim,
                                             rnn_type=args.rnn_type,
                                             ksize=args.kernel_size,
                                             n_level=args.encoder_layer,
                                             rnn_level=args.rnn_depth,
                                             nb_heads=args.nb_heads)
            dec_in = 4 * self.embed_dim
        elif args.enc_type == 5:
            self.sent_encoder = HWTransformer(max_pos_embeddings=args.max_pos_embeddings,
                                              nb_layers=args.encoder_layer,
                                              nb_heads=args.nb_heads,
                                              d_model=self.embed_dim,
                                              d_inner=args.d_ff)
            dec_in = 4 * self.embed_dim
        elif args.enc_type == 6:
            self.sent_encoder = ConvTransformer(args)
            dec_in = 4 * self.embed_dim
        else:
            raise NotImplementedError

        self.bilstm2 = LSTM(input_size=dec_in, hidden_size=args.hidden_size, bidirectional=True)

        ACT = nn.Tanh()
        # ACT = nn.LeakyReLU(0.1)
        # ACT = nn.ELU()
        self.ffn = nn.Sequential(
                                 # nn.BatchNorm1d(8*args.hidden_size),
                                 nn.Linear(in_features=8*args.hidden_size, out_features=args.linear_size),
                                 ACT,
                                 # nn.BatchNorm1d(args.linear_size),
                                 Dropout(args.dropout),
                                 nn.Linear(in_features=args.linear_size, out_features=args.linear_size),
                                 ACT,
                                 # nn.BatchNorm1d(args.linear_size),
                                 Dropout(args.dropout),
                                 nn.Linear(in_features=args.linear_size, out_features=args.lbl_size))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.word_embed.weight)

    def input_encode(self, wds, ext_wds, sent_encoder):
        mask = wds.ne(0)
        sent_embed = self.word_embed(wds)
        sent_embed_ = sent_embed + self.pre_word_embed(ext_wds)
        # sent_embed_ = self.bn_embed(sent_embed.transpose(1, 2).contiguous()).transpose(1, 2)
        sent_enc_out = sent_encoder(self.embed_drop(sent_embed_), mask)
        return sent_enc_out

    def soft_align_attention(self, x1, x2, mask1=None, mask2=None):
        '''
         x1: bz * seq_len1 * hidden_size
         x2: bz * seq_len2 * hidden_size

         mask1: torch.uint8 bz * seq_len1
         mask2: torch.uint8 bz * seq_len2
         1对应pad部分，0对应有效部分
        '''
        # bz * seq_len1 * seq_len2
        att = torch.matmul(x1, x2.transpose(1, 2))
        mask1 = mask1.float().masked_fill(mask1, -1e9)
        mask2 = mask2.float().masked_fill(mask2, -1e9)
        # bz * seq_len1 * seq_len2
        weight1 = F.softmax(att + mask2.unsqueeze(1), dim=-1)
        # bz * seq_len1 * hidden_size
        x1_align = torch.matmul(weight1, x2)
        # bz * seq_len2 * seq_len1
        weight2 = F.softmax(att.transpose(1, 2) + mask1.unsqueeze(1), dim=-1)
        # bz * seq_len2 * hidden_size
        x2_align = torch.matmul(weight2, x1)
        return x1_align, x2_align

    def sub_mul(self, x1, x2):
        mul = x1 * x2
        sub = x1 - x2
        return torch.cat((sub, mul), -1)

    def pooling(self, x):
        avg_pool = F.avg_pool1d(x.transpose(1, 2), x.size(1)).squeeze(-1)
        max_pool = F.max_pool1d(x.transpose(1, 2), x.size(1)).squeeze(-1)
        return torch.cat((avg_pool, max_pool), -1)

    def forward(self, *inputs):
        '''
        :param inputs:
            (sent1_idx, sent2_idx),
            (ext_sent1_idx, ext_sent2_idx)
        :return:
        '''
        # input encoding
        sent1, sent2 = inputs[0]
        ext_sent1, ext_sent2 = inputs[1]
        pad_mask1, pad_mask2 = sent1.eq(0), sent2.eq(0)
        s1_enc = self.input_encode(sent1, ext_sent1, self.sent_encoder)
        s2_enc = self.input_encode(sent2, ext_sent2, self.sent_encoder)

        # local inference modeling
        s1_align, s2_align = self.soft_align_attention(s1_enc, s2_enc, pad_mask1, pad_mask2)
        s1_combined = torch.cat((s1_enc, s1_align, self.sub_mul(s1_enc, s1_align)), dim=-1)
        s2_combined = torch.cat((s2_enc, s2_align, self.sub_mul(s2_enc, s2_align)), dim=-1)

        # inference composition
        s1_compose = self.bilstm2(s1_combined, ~pad_mask1)
        s2_compose = self.bilstm2(s2_combined, ~pad_mask2)
        s1_out = self.pooling(s1_compose)
        s2_out = self.pooling(s2_compose)

        sim = self.ffn(torch.cat((s1_out, s2_out), dim=-1))
        return sim

