import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualConv(nn.Module):
    def __init__(self, conv_layer, dropout=0.2):
        super(ResidualConv, self).__init__()
        assert isinstance(conv_layer, nn.Conv1d)
        self.conv_layer = conv_layer
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs):
        return inputs + self.dropout(self.conv_layer(inputs))


class ConvEncoder(nn.Module):
    def __init__(self, in_features, out_features, num_convs, kernel_size, dropout=0.2):
        super(ConvEncoder, self).__init__()

        assert in_features == out_features  # 如果输入输出维度不一致，则无法使用残差连接
        # 保证卷积前后序列长度不变：kernel_size = 2 * pad + 1
        assert kernel_size % 2 == 1
        padding = kernel_size // 2

        self.conv_layers = nn.Sequential()
        for i in range(num_convs):
            rconv = ResidualConv(nn.Conv1d(in_channels=in_features,
                                           out_channels=out_features,
                                           kernel_size=kernel_size,
                                           padding=padding), dropout)
            self.conv_layers.add_module(name=f'conv_{i}', module=rconv)

            self.conv_layers.add_module(name=f'activation_{i}', module=nn.ReLU())

    def forward(self, embed_inp, non_pad_mask=None):
        '''
        :param embed_inp: (bz, seq_len, embed_dim)
        :param non_pad_mask: (bz, seq_len)
        :return:
        '''
        if non_pad_mask is not None:
            # (bz, seq_len, embed_dim) * (bz, seq_len, 1)  广播
            embed_inp *= non_pad_mask.unsqueeze(dim=-1)

        conv_out = self.conv_layers(embed_inp.transpose(1, 2)).transpose(1, 2)
        return conv_out


# 提取k个最大值并保持相对顺序不变
class KMaxPool1d(nn.Module):
    def __init__(self, top_k: int):
        super(KMaxPool1d, self).__init__()
        self.top_k = top_k

    def forward(self, inputs):
        assert inputs.dim() == 3
        # torch.topk和torch.sort均返回的是：values, indices
        top_idxs = torch.topk(inputs, k=self.top_k, dim=2)[1]
        sorted_top_idxs = top_idxs.sort(dim=2)[0]
        # gather: 沿给定轴dim, 将输入索引张量index指定位置的值进行聚合(抽取)
        return inputs.gather(dim=2, index=sorted_top_idxs)


class ConvStack(nn.Module):
    def __init__(self, in_features, out_features, num_convs=3, filter_size=100, kernel_sizes=(1, 3, 5), dropout=0.1):
        super(ConvStack, self).__init__()

        self.conv_stack = nn.Sequential()
        for i in range(num_convs):
            conv_i = ConvEncoder(in_features, out_features, filter_size, kernel_sizes, dropout)
            self.conv_stack.add_module(f'conv_{i}', conv_i)
            self.conv_stack.add_module(f'activate_{i}', nn.ReLU())

    def forward(self, inputs):
        return self.conv_stack(inputs)


class LightweightConv1d(nn.Module):
    def __init__(self, input_size, kernel_size=1, padding=0, num_heads=1,
                 weight_softmax=False, bias=False, weight_dropout=0.1):
        super(LightweightConv1d, self).__init__()
        self.input_size = input_size
        self.kernel_size = kernel_size
        self.num_heads = num_heads
        self.padding = padding
        self.weight_softmax = weight_softmax
        self.weight = nn.Parameter(torch.rand(num_heads, 1, kernel_size))

        if self.kernel_size % 2 == 0:
            # 偶数卷积核，在左右添加pad
            self.align_padding = nn.ZeroPad2d(padding=(self.padding[0], self.padding[1], 0, 0))

        if bias:
            self.bias = nn.Parameter(torch.zeros(input_size))
        else:
            self.bias = None
        self.weight_dropout = weight_dropout
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            nn.init.constant_(self.bias, 0.)

    def forward(self, inp):
        '''
        input size: B x T x C
        output size: B x T x C
        '''
        inp = inp.transpose(1, 2)
        B, C, T = inp.size()
        H = self.num_heads

        weight = self.weight
        if self.weight_softmax:
            weight = F.softmax(weight, dim=-1)

        weight = F.dropout(weight, self.weight_dropout, training=self.training)
        # Merge every C/H entries into the batch dimension (C = self.input_size)
        # B x C x T -> (B * C/H) x H x T
        inp = inp.reshape(-1, H, T)
        if self.kernel_size % 2 == 0:
            inp = self.align_padding(inp)
            output = F.conv1d(inp, weight, groups=self.num_heads)
        else:
            output = F.conv1d(inp, weight, padding=self.padding, groups=self.num_heads)

        output = output.reshape(B, C, T).transpose(1, 2)
        if self.bias is not None:
            output = output + self.bias.reshape(1, -1, 1)

        return output
