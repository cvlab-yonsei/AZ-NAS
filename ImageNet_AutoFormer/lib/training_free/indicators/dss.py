import torch

from . import indicator
from ..p_utils import get_layer_metric_array_dss
import torch.nn as nn

@indicator('dss', bn=False, mode='param')
def compute_dss_per_weight(net, inputs, targets, mode, split_data=1, loss_fn=None):

    device = inputs.device

    @torch.no_grad()
    def linearize(net):
        signs = {}
        for name, param in net.state_dict().items():
            signs[name] = torch.sign(param)
            param.abs_()
        return signs

    @torch.no_grad()
    def nonlinearize(net, signs):
        for name, param in net.state_dict().items():
            if 'weight_mask' not in name:
                param.mul_(signs[name])

    signs = linearize(net)

    net.zero_grad()
    input_dim = list(inputs[0,:].shape)
    inputs = torch.ones([1] + input_dim).float().to(device)
    output = net.forward(inputs)
    torch.sum(output).backward()

    def dss(layer):
        if layer._get_name() == 'PatchembedSuper':
            if layer.sampled_weight.grad is not None:
                return torch.abs(layer.sampled_weight.grad * layer.sampled_weight)
            else:
                return torch.zeros_like(layer.sampled_weight)
        if isinstance(layer, nn.Linear) and 'qkv' in layer._get_name() and layer.samples or isinstance(layer,
                                                                                                       nn.Linear) and layer.out_features == layer.in_features and layer.samples:
            if layer.samples['weight'].grad is not None:
                return torch.abs(
                    torch.norm(layer.samples['weight'].grad, 'nuc') * torch.norm(layer.samples['weight'], 'nuc'))
            else:
                return torch.zeros_like(layer.samples['weight'])
        if isinstance(layer,
                      nn.Linear) and 'qkv' not in layer._get_name() and layer.out_features != layer.in_features and layer.out_features != 1000 and layer.samples:
            if layer.samples['weight'].grad is not None:
                return torch.abs(layer.samples['weight'].grad * layer.samples['weight'])
            else:
                return torch.zeros_like(layer.samples['weight'])
        elif isinstance(layer, torch.nn.Linear) and layer.out_features == 1000:
            if layer.weight.grad is not None:
                return torch.abs(layer.weight.grad * layer.weight)
            else:
                return torch.zeros_like(layer.weight)
        else:
            return torch.tensor(0).to(device)
    grads_abs = get_layer_metric_array_dss(net, dss, mode)

    nonlinearize(net, signs)

    return grads_abs


