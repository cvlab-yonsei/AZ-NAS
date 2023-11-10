import os, sys
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from torch import nn
import numpy as np

from model.module.Linear_super import LinearSuper
from model.module.embedding_super import PatchembedSuper
from model.module.qkv_super import qkv_super

def compute_nas_score(model, device, trainloader, resolution, batch_size):
    model.eval() # eval mode
    info = {}

    if trainloader == None:
        input_ = torch.randn(size=[batch_size, 3, resolution, resolution], device=device)
    else:
        input_ = next(iter(trainloader))[0]
        input_ = input_.to(device)
    
    res_features = model.extract_res_features(input_)

    ################ expressivity score ################
    expressivity_scores = []
    for i in range(len(res_features)):
        feat = res_features[i].detach().clone()
        b,n,c = feat.size()
        feat = feat.view(b*n,c)
        m = feat.mean(dim=0, keepdim=True)
        feat = feat - m
        sigma = torch.mm(feat.transpose(1,0),feat) / (feat.size(0))
        s = torch.linalg.eigvalsh(sigma) # faster version for computing eignevalues, can be adopted since sigma is symmetric
        prob_s = s / s.sum()
        score = (-prob_s)*torch.log(prob_s+1e-8)
        score = score.sum().item()
        expressivity_scores.append(score)
    expressivity_scores = np.array(expressivity_scores)
    expressivity = np.sum(expressivity_scores)
    #################################################

    ################ trainability score ##############
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
    trainability = np.mean(scores)
    #################################################

    info['expressivity'] = float(expressivity) if not np.isnan(expressivity) else -np.inf
    info['trainability'] = float(trainability) if not np.isnan(trainability) else -np.inf
    info['complexity'] = float(model.get_complexity(model.patch_embed_super.num_patches))

    return info