import torch.nn as nn


def count_flops(model, subnet_m=None, subnet_c=None, input_shape=[3, 224, 224], heads_share=False):
    flops = []
    c, w, h = input_shape
    for m in model.modules():
        # Embedding layer
        if isinstance(m, nn.Conv2d):
            w = (w + m.padding[0] * 2 - m.kernel_size[0]) // m.stride[0] + 1
            h = (h + m.padding[1] * 2 - m.kernel_size[1]) // m.stride[1] + 1
            c_in = m.out_channels

        elif isinstance(m, nn.Linear) and 'qkv' not in m._get_name() and m.out_features != m.in_features and m.out_features != 1000 and m.samples:
            flops.append(m.get_complexity(h*w))
            c_in = m.sample_out_dim
        elif 'AttentionSuper' in m._get_name() and m.qkv.samples:
            flops.append(m.get_complexity(h*w))
            c_in = m.proj.sample_out_dim
        elif isinstance(m, nn.Linear) and m.out_features == 1000:
            flops.append(m.get_complexity(h*w))
            c_in = m.sample_out_dim
    return sum(flops)
