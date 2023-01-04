import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np

__all__ = ['QConv']

class STE_round(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x_in):
        x_out = torch.round(x_in)
        return x_out
    @staticmethod
    def backward(ctx, g):
        return g

# ref. https://github.com/hustzxd/LSQuantization/blob/master/lsq.py
class QConv(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, symmetric_act=False):
        super(QConv, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.symmetric_act = symmetric_act
        
        self.STE_round = STE_round.apply
        self.bit_weight = 1
        self.bit_act = 1

        self.Qn_w = -2**(self.bit_weight-1) # does not used for 1-bit
        self.Qp_w = 2**(self.bit_weight-1) - 1 # does not used for 1-bit
        if not self.symmetric_act:
            self.Qn_a = 0
            self.Qp_a = 2**(self.bit_act) - 1
        else:
            self.Qn_a = -2**(self.bit_act-1) # does not used for 1-bit
            self.Qp_a = 2**(self.bit_act-1) - 1 # does not used for 1-bit
        
        self.sW = nn.Parameter(data = torch.tensor(1).float())
        self.sA = nn.Parameter(data = torch.tensor(1).float())

        self.register_buffer('init', torch.tensor([0]))
        self.register_buffer('search', torch.tensor([1]))

    def set_q_range(self, bit_weight, bit_act):
        self.bit_weight = bit_weight
        self.bit_act = bit_act

        self.Qn_w = -2**(self.bit_weight-1) # does not used for 1-bit
        self.Qp_w = 2**(self.bit_weight-1) - 1 # does not used for 1-bit
        if not self.symmetric_act:
            self.Qn_a = 0
            self.Qp_a = 2**(self.bit_act) - 1
        else:
            self.Qn_a = -2**(self.bit_act-1) # does not used for 1-bit
            self.Qp_a = 2**(self.bit_act-1) - 1 # does not used for 1-bit

    def weight_quantization(self, weight):
        if self.bit_weight == 32:
            return weight
        elif self.bit_weight == 1:
            weight = weight / (torch.abs(self.sW)+1e-6)
            weight = weight.clamp(-1, 1) # [-1, 1]
            weight = (weight + 1)/2 # [0, 1]
            weight = self.STE_round(weight) # {0, 1}
            weight = weight * 2 - 1 # {-1, 1}
            return weight
        else:
            weight = weight / (torch.abs(self.sW)+1e-6) # normalized such that 99% of weights lie in [-1, 1]
            weight = weight * 2**(self.bit_weight-1)
            weight = weight.clamp(self.Qn_w, self.Qp_w)
            weight = self.STE_round(weight) # {-2^(b-1), ..., 2^(b-1)-1}
            weight = weight / 2**(self.bit_weight-1) # fixed point representation
            return weight

    def act_quantization(self, x):
        if self.bit_act == 32:
            return x
        if not self.symmetric_act:
            if self.bit_act == 1:
                x = x / (torch.abs(self.sA)+1e-6)
                x = x.clamp(0, 1) # [0, 1]
                x = self.STE_round(x) # {0, 1}
                return x
            else:
                x = x / (torch.abs(self.sA)+1e-6) # normalized such that 99% of activations lie in [0, 1]
                x = x * 2**self.bit_act
                x = x.clamp(self.Qn_a, self.Qp_a) # [0, 2^b-1]
                x = self.STE_round(x) # {0, ..., 2^b-1}
                x = x / 2**self.bit_act # fixed point representation
                return x
        else:
            if self.bit_act == 1:
                x = x / (torch.abs(self.sA)+1e-6)
                x = x.clamp(-1, 1) # [-1, 1]
                x = (x+1) / 2 # [0, 1]
                x = self.STE_round(x) # {0, 1}
                x = x * 2 - 1 # {-1, 1}
                return x
            else:
                x = x / (torch.abs(self.sA)+1e-6)
                x = x * 2**(self.bit_act-1)
                x = x.clamp(self.Qn_a, self.Qp_a)
                x = self.STE_round(x) # {-2^(b-1), ..., 2^(b-1)-1}
                x = x / 2**(self.bit_act-1)
                return x


    def initialize(self, x):
        self.sW.data.fill_(self.weight.std()*3.0)
        if not self.symmetric_act:
            self.sA.data.fill_(x.std() / math.sqrt(1 - 2/math.pi) * 3.0)
        else:
            self.sA.data.fill_(x.std() * 3.0)
        # self.init.fill_(0)

    def forward(self, x):
        if self.training and self.init:
            self.initialize(x)

        if self.search:
            ## weight
            k = int(0.01 * self.weight.numel())
            if k==0:
                k = 1
            sW = self.weight.abs().view(-1).topk(k)[0].min()
            self.sW.data.fill_(sW)

            ## act
            k = int(0.01 * x.numel())
            if k==0:
                k = 1
            sA = x.abs().view(-1).topk(k)[0].min()
            self.sA.data.fill_(sA)
       
        Qweight = self.weight_quantization(self.weight)
        Qact = self.act_quantization(x)
        output = F.conv2d(Qact, Qweight, self.bias,  self.stride, self.padding, self.dilation, self.groups)

        return output