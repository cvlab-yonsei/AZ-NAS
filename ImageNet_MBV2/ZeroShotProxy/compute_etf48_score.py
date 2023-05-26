import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from torch import nn
import numpy as np
import global_utils, argparse, ModelLoader, time

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
    # elif method == 'kaiming_uni_fanin':
    #     model.apply(kaiming_uniform_fanin_init)
    # elif method == 'kaiming_uni_fanout':
    #     model.apply(kaiming_uniform_fanout_init)
    # elif method == 'xavier_norm':
    #     model.apply(xavier_normal)
    # elif method == 'xavier_uni':
    #     model.apply(xavier_uniform)
    # elif method == 'plain_norm':
    #     model.apply(plain_normal)
    # else:
    #     raise NotImplementedError
    return model

def cross_entropy(logit, target):
    # target must be one-hot format!!
    prob_logit = torch.nn.functional.log_softmax(logit, dim=1)
    loss = -(target * prob_logit).sum(dim=1).mean()
    return loss


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
    
    if model.no_reslink:
        layer_features = model.extract_layer_features_nores(input_)
    else:
        layer_features, output = model.extract_layer_features_and_logit(input_)
    
    ################ fwrd pca score ################
    """
    pca score across residual block features / normalize each score by upper bound
    """
    with torch.no_grad():
        scores = []
        for i in range(len(layer_features)):
            feat = layer_features[i].detach().clone()
            b,c,h,w = feat.size()
            feat = feat.permute(0,2,3,1).contiguous().view(b*h*w,c)
            m = feat.mean(dim=0, keepdim=True)
            feat = feat - m
            sigma = torch.mm(feat.transpose(1,0),feat) / (feat.size(0))
            # s = torch.linalg.svdvals(sigma) # compute signluar values (the same as eigenvalues for a symmetric mat)
            s = torch.linalg.eigvalsh(sigma) # faster version for computing eignevalues, can be adopted since sigma is symmetric
            prob_s = s / s.sum()
            score = (-prob_s)*torch.log(prob_s+1e-8)
            score = score.sum().item()
            scores.append(score / np.log(c)) # normalize by an upper bound (= np.log(c))
        fwrd_pca_score = np.mean(scores)
    #################################################

    # ################ fwrd norm score ################
    # """
    # residual block features
    # """
    # with torch.no_grad():
    #     scores = []
    #     for i in range(1, len(layer_features)):
    #         f_out = layer_features[i]
    #         f_in = layer_features[i-1]

    #         if (f_out.size() == f_in.size()) and (torch.all(f_in == f_out)):
    #             scores.append(-np.inf)
    #         else:
    #             s = f_out.abs().mean() / (f_in.abs().mean()+1e-6)
    #             scores.append(-s.item() - 1/(s.item()+1e-6) + 2)
    #     fwrd_norm_score = np.mean(scores)
    # #################################################

    # #################################################
    # """
    # angular diff std
    # """
    model.zero_grad(set_to_none=True)
    
    num_classes = output.shape[1]
    y = torch.randint(low=0, high=num_classes, size=[batch_size], device=device)
    one_hot_y = torch.nn.functional.one_hot(y, num_classes).float()
    loss = cross_entropy(output, one_hot_y)
    loss.backward()

    # scores = []
    # with torch.no_grad():
    #     for p in model.parameters():
    #         if hasattr(p, 'grad') and p.grad is not None:
    #             if len(p.size()) == 4: # conv weights
    #                 g_norm = torch.norm(p.grad).item()
    #                 p_norm = torch.norm(p).item()
    #                 scores.append(g_norm/p_norm)
    # trainability = np.mean(scores)
    # #################################################

    ################ bkwd norm score ##############
    """
    grad stability score across residual block features
    """
    scores = []
    for i in reversed(range(1, len(layer_features))):
        g_out = layer_features[i].grad.detach().clone()
        g_in = layer_features[i-1].grad.detach().clone()

        if g_out.size()==g_in.size() and torch.all(g_in == g_out):
            scores.append(-np.inf)
        else:
            s = g_out.abs().mean() / (g_in.abs().mean()+1e-6)
            scores.append(-s.item() - 1/(s.item()+1e-6) + 2)
    bkwd_norm_score = np.mean(scores)
    #################################################

    info['expressivity'] = float(fwrd_pca_score) if not np.isnan(fwrd_pca_score) else -np.inf
    # info['fwrd_stability'] = float(fwrd_norm_score) if not np.isnan(fwrd_norm_score) else -np.inf
    info['bkwd_stability'] = float(bkwd_norm_score) if not np.isnan(bkwd_norm_score) else -np.inf
    # info['trainability'] = float(trainability) if not np.isnan(trainability) else -np.inf
    info['capacity'] = float(model.get_model_size())
    info['complexity'] = float(model.get_FLOPs(resolution))
    return info


def parse_cmd_options(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=16, help='number of instances in one mini-batch.')
    parser.add_argument('--input_image_size', type=int, default=None,
                        help='resolution of input image, usually 32 for CIFAR and 224 for ImageNet.')
    parser.add_argument('--repeat_times', type=int, default=32)
    parser.add_argument('--gpu', type=int, default=None)
    parser.add_argument('--mixup_gamma', type=float, default=1e-2)
    module_opt, _ = parser.parse_known_args(argv)
    return module_opt

if __name__ == "__main__":
    opt = global_utils.parse_cmd_options(sys.argv)
    args = parse_cmd_options(sys.argv)
    the_model = ModelLoader.get_model(opt, sys.argv)
    if args.gpu is not None:
        the_model = the_model.cuda(args.gpu)


    start_timer = time.time()
    info = compute_nas_score(gpu=args.gpu, model=the_model,
                             resolution=args.input_image_size, batch_size=args.batch_size, fp16=False)
    time_cost = (time.time() - start_timer)
    expressivity = info['expressivity']
    stability = info['stability']
    trainability = info['trainability']
    capacity = info['capacity']
    print(f'Expressivity={expressivity:.4g}, Stability={stability:.4g}, Trainability={trainability:.4g}, Capacity={capacity:.4g}, time cost={time_cost:.4g} second(s)')