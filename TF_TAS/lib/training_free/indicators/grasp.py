import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd

from . import indicator
from ..p_utils import get_layer_metric_array


@indicator('grasp', bn=True, mode='param')
def compute_grasp_per_weight(net, inputs, targets, mode, loss_fn, T=1, num_iters=1, split_data=1):
    # get all applicable weights
    weights = []
    for layer in net.modules():
        if layer._get_name() == 'PatchembedSuper':
            weights.append(layer.sampled_weight)
            layer.sampled_weight.requires_grad_(True)  # TODO isn't this already true?
        if isinstance(layer, nn.Linear) and layer.out_features != 1000 and layer.samples:
            weights.append(layer.samples['weight'])
            layer.samples['weight'].requires_grad_(True)  # TODO isn't this already true?
        if isinstance(layer, torch.nn.Linear) and layer.out_features == 1000:
            weights.append(layer.samples['weight'])
            layer.weight.requires_grad_(True)  # TODO isn't this already true?

    # NOTE original code had some input/target splitting into 2
    # I am guessing this was because of GPU mem limit
    net.zero_grad()
    N = inputs.shape[0]
    for sp in range(split_data):
        st = sp * N // split_data
        en = (sp + 1) * N // split_data

        # forward/grad pass #1
        grad_w = None
        for _ in range(num_iters):
            # TODO get new data, otherwise num_iters is useless!
            outputs = net.forward(inputs[st:en]) / T
            loss = loss_fn(outputs, targets[st:en])
            grad_w_p = autograd.grad(loss, weights, allow_unused=True)
            if grad_w is None:
                grad_w = list(grad_w_p)
            else:
                for idx in range(len(grad_w)):
                    grad_w[idx] += grad_w_p[idx]

    for sp in range(split_data):
        st = sp * N // split_data
        en = (sp + 1) * N // split_data

        # forward/grad pass #2
        outputs = net.forward(inputs[st:en]) / T
        loss = loss_fn(outputs, targets[st:en])
        grad_f = autograd.grad(loss, weights, create_graph=True, allow_unused=True)

        # accumulate gradients computed in previous step and call backwards
        z, count = 0, 0
        for layer in net.modules():
            if layer._get_name() == 'PatchembedSuper':
                if grad_w[count] is not None:
                    z += (grad_w[count].data * grad_f[count]).sum()
                count += 1
            if isinstance(layer, nn.Linear) and layer.out_features != 1000 and layer.samples:
                if grad_w[count] is not None:
                    z += (grad_w[count].data * grad_f[count]).sum()
                count += 1
            if isinstance(layer, nn.Linear) and layer.out_features == 1000:
                if grad_w[count] is not None:
                    z += (grad_w[count].data * grad_f[count]).sum()
                count += 1
        z.backward()

    # compute final sensitivity metric and put in grads
    def grasp(layer):
        if layer._get_name() == 'PatchembedSuper':
            if layer.sampled_weight.grad is not None:
                return -layer.sampled_weight.data * layer.sampled_weight.grad  # -theta_q Hg
                # NOTE in the grasp code they take the *bottom* (1-p)% of values
                # but we take the *top* (1-p)%, therefore we remove the -ve sign
                # EDIT accuracy seems to be negatively correlated with this metric, so we add -ve sign here!
            else:
                return torch.zeros_like(layer.sampled_weight)
        if isinstance(layer, nn.Linear) and layer.out_features != 1000 and layer.samples:
            if layer.samples['weight'].grad is not None:
                return -layer.samples['weight'].data * layer.samples['weight'].grad  # -theta_q Hg
                # NOTE in the grasp code they take the *bottom* (1-p)% of values
                # but we take the *top* (1-p)%, therefore we remove the -ve sign
                # EDIT accuracy seems to be negatively correlated with this metric, so we add -ve sign here!
            else:
                return torch.zeros_like(layer.samples['weight'])
        if isinstance(layer, torch.nn.Linear) and layer.out_features == 1000:
            if layer.samples['weight'].grad is not None:
                return -layer.samples['weight'].data * layer.samples['weight'].grad  # -theta_q Hg
                # NOTE in the grasp code they take the *bottom* (1-p)% of values
                # but we take the *top* (1-p)%, therefore we remove the -ve sign
                # EDIT accuracy seems to be negatively correlated with this metric, so we add -ve sign here!
            else:
                return torch.zeros_like(layer.samples['weight'])

    grads = get_layer_metric_array(net, grasp, mode)

    return grads