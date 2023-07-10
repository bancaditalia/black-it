import numpy as np

import torch.nn as nn
from torch.nn.utils import weight_norm


class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()

class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding):
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1,
                                 self.conv2, self.chomp2, self.relu2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs: int, num_outputs: int, num_channels: int, 
                 kernel_size: int = 32, summary_size: int = 10):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size-1) * dilation_size)]

        self.network = nn.Sequential(
            *layers, 
            nn.Conv1d(num_channels[-1], num_outputs, summary_size),
            # Flatten
            nn.Flatten(),
            nn.LazyLinear(num_outputs) # TODO: Compute input size instead of using LazyLinear

        )

    def forward(self, x):
        return self.network(x)
    

def necessary_layers(
        dilation: int, kernel_size: int, receptive_field_size: int,
        use_ceil: bool = False, use_floor: bool = False
) -> float:
    """
    Compute the number of layers necessary to achieve a given receptive field 
    size.
    """
    
    if use_ceil and use_floor:
        raise ValueError("use_ceil and use_floor cannot both be True")
    
    n = np.emath.logn(dilation, 
            (receptive_field_size - 1) * (dilation - 1) / (kernel_size - 1)) + 1
    if use_ceil:
        return np.ceil(n)
    elif use_floor:
        return np.floor(n)
    else:
        return n