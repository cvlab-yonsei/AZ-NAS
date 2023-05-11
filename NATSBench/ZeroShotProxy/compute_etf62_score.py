import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from torch import nn
import numpy as np

def kaiming_normal_fanin_init(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
        if hasattr(m, 'bias') and m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
        if m.affine:
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

def kaiming_normal_fanout_init(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if hasattr(m, 'bias') and m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
        if m.affine:
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

def init_model(model, method='kaiming_norm_fanin'):
    if method == 'kaiming_norm_fanin':
        model.apply(kaiming_normal_fanin_init)
    elif method == 'kaiming_norm_fanout':
        model.apply(kaiming_normal_fanout_init)
    else:
        raise NotImplementedError
    return model


def compute_nas_score(model, gpu, trainloader, resolution, batch_size, fp16=False):
    model.train()
    model.cuda()
    info = {}
    nas_score_list = []
    if gpu is not None:
        device = torch.device('cuda:{}'.format(gpu))
    else:
        device = torch.device('cpu')

    if fp16:
        dtype = torch.half
    else:
        dtype = torch.float32

    init_model(model, 'kaiming_norm_fanin')

    if trainloader == None:
        input_ = torch.randn(size=[batch_size, 3, resolution, resolution], device=device, dtype=dtype)
    else:
        input_ = next(iter(trainloader))[0]
    
    layer_features = model.extract_cell_features(input_)

    ################ fwrd pca score ################
    """
    pca score across residual block features / normalize each score by upper bound
    """
    info_flow_scores = []
    expressivity_scores = []
    for i in range(len(layer_features)):
        feat = layer_features[i].detach().clone()
        b,c,h,w = feat.size()
        feat = feat.permute(0,2,3,1).contiguous().view(b*h*w,c)
        m = feat.mean(dim=0, keepdim=True)
        feat = feat - m
        sigma = torch.mm(feat.transpose(1,0),feat) / (feat.size(0))
        s = torch.linalg.eigvalsh(sigma) # faster version for computing eignevalues, can be adopted since sigma is symmetric
        prob_s = s / s.sum()
        score = (-prob_s)*torch.log(prob_s+1e-8)
        score = score.sum().item()
        info_flow_scores.append(score) 
        expressivity_scores.append(score) # normalize by an upper bound (= np.log(c))
    info_flow_scores = np.array(info_flow_scores)
    info_flow = np.min(info_flow_scores[1:] - info_flow_scores[:-1])
    expressivity = np.mean(expressivity_scores)
    #################################################

    # ################ fwrd norm score ################
    # """
    # L2 norm preserving after matching the resolution / residual block features
    # """
    # scores = []
    # for i in range(1, len(layer_features)):
    #     f_out = layer_features[i]
    #     f_in = layer_features[i-1]

    #     if (f_out.size() == f_in.size()) and (torch.all(f_in == f_out)):
    #         scores.append(-np.inf)
    #     else:
    #         # if f_out.size(2) != f_in.size(2) or f_out.size(3) != f_in.size(3):
    #         #     bo,co,ho,wo = f_out.size()
    #         #     bi,ci,hi,wi = f_in.size()
    #         #     stride = int(hi/ho)
    #         #     pixel_unshuffle = nn.PixelUnshuffle(stride)
    #         #     f_in = pixel_unshuffle(f_in)
    #         s = f_out.norm(p=2, dim=(1)).mean() / (f_in.norm(p=2, dim=(1)).mean()+1e-6)
    #         scores.append(-s.item() - 1/(s.item()+1e-6) + 2)
    # fwrd_norm_score = np.mean(scores)
    # #################################################

    ################ spec norm score ##############
    """
    spec norm score across residual block features
    """
    scores = []
    for i in reversed(range(1, len(layer_features))):
        f_out = layer_features[i]
        f_in = layer_features[i-1]
        if f_out.grad is not None:
            f_out.grad.zero_()
        if f_in.grad is not None:
            f_in.grad.zero_()
        
        g_out = torch.ones_like(f_out) * 0.5
        g_out = (torch.bernoulli(g_out) - 0.5) * 2
        g_in = torch.autograd.grad(outputs=f_out, inputs=f_in, grad_outputs=g_out, retain_graph=False)[0]
        if g_out.size()==g_in.size() and torch.all(g_in == g_out):
            scores.append(-np.inf)
        else:
            if g_out.size(2) != g_in.size(2) or g_out.size(3) != g_in.size(3):
                bo,co,ho,wo = g_out.size()
                bi,ci,hi,wi = g_in.size()
                stride = int(hi/ho)
                pixel_unshuffle = nn.PixelUnshuffle(stride)
                g_in = pixel_unshuffle(g_in)
            bo,co,ho,wo = g_out.size()
            bi,ci,hi,wi = g_in.size()
            ### straight-forward way
            # g_out = g_out.permute(0,2,3,1).contiguous().view(bo*ho*wo,1,co)
            # g_in = g_in.permute(0,2,3,1).contiguous().view(bi*hi*wi,ci,1)
            # mat = torch.bmm(g_in,g_out).mean(dim=0)
            ### efficient way # print(torch.allclose(mat, mat2, atol=1e-6))
            g_out = g_out.permute(0,2,3,1).contiguous().view(bo*ho*wo,co)
            g_in = g_in.permute(0,2,3,1).contiguous().view(bi*hi*wi,ci)
            mat = torch.mm(g_in.transpose(1,0),g_out) / (bo*ho*wo)
            ### make faster on cpu
            if mat.size(0) < mat.size(1):
                mat = mat.transpose(0,1)
            ###
            s = torch.linalg.svdvals(mat)
            scores.append(-s.max().item() - 1/(s.max().item()+1e-6)+2)
    bkwd_norm_score = np.mean(scores)
    #################################################

    info['expressivity'] = float(expressivity) if not np.isnan(expressivity) else -np.inf
    info['info_flow'] = float(info_flow) if not np.isnan(info_flow) else -np.inf
    # info['stability'] = float(fwrd_norm_score) if not np.isnan(fwrd_norm_score) else -np.inf
    info['trainability'] = float(bkwd_norm_score) if not np.isnan(bkwd_norm_score) else -np.inf
    # info['capacity'] = float(model.get_model_size())
    # info['complexity'] = float(model.get_FLOPs(resolution))
    return info