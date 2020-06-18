import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import torch.nn.functional as F
from torch.nn.modules.utils import _single, _pair, _triple


class attention_net(nn.Module):
    def __init__(self, in_channels, r=30, k=4):
        super(attention_net, self).__init__()
        self.in_channels = in_channels
        self.r = r
        self.k = k
        self.fc1 = nn.Linear(self.in_channels, self.in_channels // 4)
        self.fc2 = nn.Linear(self.in_channels // 4, self.k)

    def forward(self, input):
        output = F.adaptive_avg_pool2d(input, (1, 1))
        output = output.view(-1, self.in_channels)
        output = F.relu(self.fc1(output))
        output = F.relu(self.fc2(output))
        output = output.view(-1, self.k) / self.r

        return F.softmax(output, dim=0)


class DyConv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1,
                 bias=True, padding_mode='zeros', k=4):
        super(DyConv2d, self).__init__(in_channels, out_channels, kernel_size, stride,
                                       padding, dilation, groups,
                                       bias, padding_mode)
        self.k = k
        self.attention_net = attention_net(in_channels, r=30, k=4)
        self.weight = Parameter(torch.Tensor(k*self.out_channels, self.in_channels // self.groups, *self.kernel_size))
        if bias:
            self.bias = Parameter(torch.Tensor(k*out_channels))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def _conv_forward(self, input, weight, bias):
        if self.padding_mode != 'zeros':
            return F.conv2d(F.pad(input, self._padding_repeated_twice, mode=self.padding_mode),
                            weight, bias, self.stride,
                            _pair(0), self.dilation, self.groups)
        return F.conv2d(input, weight, bias, self.stride,
                        self.padding, self.dilation, self.groups)

    def forward(self, input):
        pi = self.attention_net(input)
        weight = self.weight.view(self.k, -1)
        weight = torch.matmul(pi, weight).view(-1, self.out_channels, self.in_channels // self.groups, *self.kernel_size)
        if self.bias is not None:
            bias = self.bias.view(self.k, -1)
            bias = torch.matmul(pi, bias).view(-1, self.out_channels)

        list = []
        for i in range(input.size(0)):
            list.append(self._conv_forward(input[i:i+1], weight[i], bias[i]))

        output = torch.cat(list, 0)

        return output


if __name__ == "__main__":
    net = DyConv2d(16, 3, 3)
    input = torch.randn(4, 16, 5, 5)
    print(input)
    output = net(input)
    print(output)
    print(output.size())
