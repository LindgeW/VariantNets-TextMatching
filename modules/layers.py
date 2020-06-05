import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, in_feature, out_feature, activation=None, bias=True):
        super(MLP, self).__init__()

        if activation is None:
            self.activation = lambda x: x
        else:
            assert callable(activation)
            self.activation = activation

        self.bias = bias
        self.linear = nn.Linear(in_features=in_feature,
                                out_features=out_feature,
                                bias=bias)
        self.reset_parameters()

    def reset_parameters(self):
        if self.bias:
            nn.init.zeros_(self.linear.bias)

    def forward(self, inputs):
        linear_out = self.linear(inputs)
        return self.activation(linear_out)


class Dropout(nn.Module):
    def __init__(self, p: float = 0.0):
        super(Dropout, self).__init__()
        self.p = p
        self._drop = nn.Dropout(p)

    def forward(self, x):
        if self.training and self.p > 0:
            x = self._drop(x)
        return x
