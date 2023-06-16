import os, sys
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from torch import nn
import numpy as np

from model.module.Linear_super import LinearSuper
from model.module.embedding_super import PatchembedSuper
from model.module.qkv_super import qkv_super

# def kaiming_normal_fanin_init(m):
#     if isinstance(m, LinearSuper):
#         if 'weight' in m.samples.keys():
#             nn.init.kaiming_normal_(m.samples['weight'], mode='fan_in', nonlinearity='relu')
#             if m.samples['bias'] is not None:
#                 nn.init.constant_(m.samples['bias'], 0)
#     elif isinstance(m, qkv_super):
#         if 'weight' in m.samples.keys():
#             ci, co = m.samples['weight'].size()
#             cs = co // 3
#             nn.init.kaiming_normal_(m.samples['weight'][:,:cs], mode='fan_in', nonlinearity='relu')
#             nn.init.kaiming_normal_(m.samples['weight'][:,cs:cs*2], mode='fan_in', nonlinearity='relu')
#             nn.init.kaiming_normal_(m.samples['weight'][:,cs*2:], mode='fan_in', nonlinearity='relu')
#             if m.samples['bias'] is not None:
#                 nn.init.constant_(m.samples['bias'], 0)
#     elif isinstance(m, PatchembedSuper):
#         nn.init.kaiming_normal_(m.sampled_weight, mode='fan_in', nonlinearity='relu')
#         if m.sampled_bias is not None:
#             nn.init.constant_(m.sampled_bias, 0)
#     elif isinstance(m, nn.LayerNorm):
#         nn.init.constant_(m.bias, 0)
#         nn.init.constant_(m.weight, 1.0)

# def xavier_uniform(m):
#     if isinstance(m, LinearSuper):
#         if 'weight' in m.samples.keys():
#             nn.init.xavier_uniform_(m.samples['weight'])
#             if m.samples['bias'] is not None:
#                 nn.init.constant_(m.samples['bias'], 0)
#     elif isinstance(m, qkv_super):
#         if 'weight' in m.samples.keys():
#             ci, co = m.samples['weight'].size()
#             cs = co // 3
#             nn.init.xavier_uniform_(m.samples['weight'][:,:cs])
#             nn.init.xavier_uniform_(m.samples['weight'][:,cs:cs*2])
#             nn.init.xavier_uniform_(m.samples['weight'][:,cs*2:])
#             if m.samples['bias'] is not None:
#                 nn.init.constant_(m.samples['bias'], 0)
#     elif isinstance(m, PatchembedSuper):
#         nn.init.xavier_uniform_(m.sampled_weight)
#         if m.sampled_bias is not None:
#             nn.init.constant_(m.sampled_bias, 0)
#     elif isinstance(m, nn.LayerNorm):
#         nn.init.constant_(m.bias, 0)
#         nn.init.constant_(m.weight, 1.0)

# def xavier_normal(m):
#     if isinstance(m, LinearSuper):
#         if 'weight' in m.samples.keys():
#             nn.init.xavier_normal_(m.samples['weight'])
#             if m.samples['bias'] is not None:
#                 nn.init.constant_(m.samples['bias'], 0)
#     elif isinstance(m, qkv_super):
#         if 'weight' in m.samples.keys():
#             ci, co = m.samples['weight'].size()
#             cs = co // 3
#             nn.init.xavier_normal_(m.samples['weight'][:,:cs])
#             nn.init.xavier_normal_(m.samples['weight'][:,cs:cs*2])
#             nn.init.xavier_normal_(m.samples['weight'][:,cs*2:])
#             if m.samples['bias'] is not None:
#                 nn.init.constant_(m.samples['bias'], 0)
#     elif isinstance(m, PatchembedSuper):
#         nn.init.xavier_normal_(m.sampled_weight)
#         if m.sampled_bias is not None:
#             nn.init.constant_(m.sampled_bias, 0)
#     elif isinstance(m, nn.LayerNorm):
#         nn.init.constant_(m.bias, 0)
#         nn.init.constant_(m.weight, 1.0)

# def init_model(model, method='kaiming_norm_fanin'):
#     if method == 'kaiming_norm_fanin':
#         model.apply(kaiming_normal_fanin_init)
#     elif method == 'xavier_uniform':
#         model.apply(xavier_uniform)
#     elif method == 'xavier_normal':
#         model.apply(xavier_normal)
#     # elif method == 'kaiming_norm_fanout':
#     #     model.apply(kaiming_normal_fanout_init)
#     else:
#         raise NotImplementedError
#     return model


def compute_nas_score(model, device, trainloader, resolution, batch_size):
    # init_model(model, 'kaiming_norm_fanin')
    # init_model(model, 'xavier_normal')
    
    model.eval() # eval mode
    info = {}

    if trainloader == None:
        input_ = torch.randn(size=[batch_size, 3, resolution, resolution], device=device)
    else:
        input_ = next(iter(trainloader))[0]
        input_ = input_.to(device)
    
    res_features = model.extract_res_features(input_)

    ################ spec norm score ##############
    """
    spec norm score across residual block features
    """
    scores = []
    for i in reversed(range(1, len(res_features))):
        f_out = res_features[i]
        f_in = res_features[i-1]
        if f_out.grad is not None:
            f_out.grad.zero_()
        if f_in.grad is not None:
            f_in.grad.zero_()
        
        g_out = torch.ones_like(f_out) * 0.5
        g_out = (torch.bernoulli(g_out) - 0.5) * 2
        g_in = torch.autograd.grad(outputs=f_out, inputs=f_in, grad_outputs=g_out, retain_graph=False)[0]
        if g_out.size()==g_in.size() and torch.all(g_in == g_out):
            continue
        else:
            if g_out.size(1) != g_in.size(1) or g_out.size(1) != g_in.size(1):
                raise NotImplementedError
            bo,no,co = g_out.size()
            bi,ni,ci = g_in.size()
            g_out = g_out.view(bo*no,co)
            g_in = g_in.view(bi*ni,ci)
            mat = torch.mm(g_in.transpose(1,0),g_out) / (bo*no)
            ### make faster on cpu
            if mat.size(0) < mat.size(1):
                mat = mat.transpose(0,1)
            ###
            s = torch.linalg.svdvals(mat)
            scores.append(-s.max().item() - 1/(s.max().item()+1e-6)+2)
    bkwd_norm_score = np.mean(scores)
    #################################################

    ####
    numels = []
    for name, module in model.named_modules():
        if hasattr(module, 'calc_sampled_param_num'):
            if name.split('.')[0] == 'blocks' and int(name.split('.')[1]) >= model.sample_layer_num:
                continue
            numels.append(module.calc_sampled_param_num())

    n_params = sum(numels) + model.sample_embed_dim[0] * (2 + model.patch_embed_super.num_patches)
    ####

    info['trainability'] = float(bkwd_norm_score) if not np.isnan(bkwd_norm_score) else -np.inf
    info['capacity'] = float(n_params)
    info['complexity'] = float(model.get_complexity(model.patch_embed_super.num_patches+1))

    return info