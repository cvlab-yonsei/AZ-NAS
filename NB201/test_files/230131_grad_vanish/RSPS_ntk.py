import os, sys, time, glob, random, argparse
import numpy as np
from copy import deepcopy
import torch
import torch.nn as nn

# XAutoDL 
from xautodl.config_utils import load_config, dict2config, configure2str
from xautodl.datasets import get_datasets, get_nas_search_loaders
from xautodl.procedures import (
    prepare_seed,
    prepare_logger,
    save_checkpoint,
    copy_checkpoint,
    get_optim_scheduler,
)
from xautodl.utils import get_model_infos, obtain_accuracy
from xautodl.log_utils import AverageMeter, time_string, convert_secs2time
from xautodl.models import get_cell_based_tiny_net, get_search_spaces

# NB201
from nas_201_api import NASBench201API as API

from ntk import get_ntk_n
from linear_region_counter import Linear_Region_Collector

import scipy.stats as stats

import tqdm

def kaiming_normal_fanin_init(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
        if hasattr(m, 'bias') and m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.BatchNorm2d):
        if m.affine:
            nn.init.ones_(m.weight.data)
            nn.init.constant_(m.bias.data, 0.0)


def kaiming_normal_fanout_init(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if hasattr(m, 'bias') and m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.BatchNorm2d):
        if m.affine:
            nn.init.ones_(m.weight.data)
            nn.init.constant_(m.bias.data, 0.0)


def init_model(model, method='kaiming_norm_fanin'):
    if method == 'kaiming_norm_fanin':
        model.apply(kaiming_normal_fanin_init)
    elif method == 'kaiming_norm_fanout':
        model.apply(kaiming_normal_fanout_init)
    return model

def search_find_best(xloader, network, lrc_model, n_samples):
    network.train()
    archs, ntk_scores, lr_scores = [], [], []
    
    for i in tqdm.tqdm(range(n_samples)):
        # random sampling
        arch = network.random_genotype(True)
        
        ntk_score_tmp = []
        for _ in range(3):
            
            init_model(network)
            # ntk
            score = get_ntk_n(xloader, [network], recalbn=0, train_mode=True, num_batch=1)[0]
            ntk_score_tmp.append(-score)
        ntk_score = np.mean(ntk_score_tmp)
        
#         lr_score_tmp = []
#         for _ in range(3):
#             init_model(network)
#             lrc_model.reinit(models=[network], seed=xargs.rand_seed)
#             num_linear_regions = lrc_model.forward_batch_sample()
#             lrc_model.clear()
#             score = np.mean(_linear_regions)
#             lr_score_tmp.append(score)
#         lr_score = np.mean(lr_score_tmp)
        
        archs.append(arch)
        ntk_scores.append(ntk_score)
#         lr_scores.append(lr_score)
        
    print(ntk_scores)
    asd
#     print(lr_scores)
    
    rank_ntk, rank_lr = stats.rankdata(ntk_scores), stats.rankdata(lr_score)

    rank_agg = rank_tnk + rank_lr

    best_idx = np.argmax(rank_agg)
    best_arch, best_ntk_score, best_lr_score = archs[best_idx], rank_ntk[best_idx], rank_lr[best_idx]

    return best_arch, best_ntk_score, best_lr_score


def main(xargs):
    train_data, valid_data, xshape, class_num = get_datasets(xargs.dataset, xargs.data_path, -1)
    config = load_config(xargs.config_path, {"class_num": class_num, "xshape": xshape}, logger)
    search_loader, train_loader, valid_loader = get_nas_search_loaders(train_data,
                                                            valid_data,
                                                            xargs.dataset,
                                                            "../../configs/nas-benchmark/",
                                                            (config.batch_size, config.test_batch_size),
                                                            xargs.workers)
    logger.log("||||||| {:10s} ||||||| Search-Loader-Num={:}, Valid-Loader-Num={:}, batch size={:}".format(
                xargs.dataset, len(search_loader), len(valid_loader), config.batch_size))
    logger.log("||||||| {:10s} ||||||| Config={:}".format(xargs.dataset, config))

    ## model
    search_space = get_search_spaces("cell", xargs.search_space_name)
    model_config = dict2config(
        {
            "name": "RANDOM",
            "C": xargs.channel,
            "N": xargs.num_cells,
            "max_nodes": xargs.max_nodes,
            "num_classes": class_num,
            "space": search_space,
            "affine": False,
            "track_running_stats": bool(xargs.track_running_stats),
        },
        None,
    )
    search_model = get_cell_based_tiny_net(model_config)

    if xargs.arch_nas_dataset is None:
        api = None
    else:
        api = API(xargs.arch_nas_dataset)
    logger.log("{:} create API = {:} done".format(time_string(), api))

    last_info, model_base_path, model_best_path = (
        logger.path("info"),
        logger.path("model"),
        logger.path("best"),
    )
    # network = torch.nn.DataParallel(search_model).cuda()
    network = search_model.cuda()

    ## LRC
    # lrc_model = Linear_Region_Collector(input_size=(1000, 1, 3, 3), sample_batch=3, dataset=xargs.dataset, data_path=xargs.data_path, seed=xargs.rand_seed)


    ## misc
    if last_info.exists():  # automatically resume from previous checkpoint
        logger.log(
            "=> loading checkpoint of the last-info '{:}' start".format(last_info)
        )
        last_info = torch.load(last_info)
        start_epoch = last_info["epoch"]
        checkpoint = torch.load(last_info["last_checkpoint"])
        genotypes = checkpoint["genotypes"]
        valid_accuracies = checkpoint["valid_accuracies"]
        search_model.load_state_dict(checkpoint["search_model"])
        logger.log(
            "=> loading checkpoint of the last-info '{:}' start with {:}-th epoch.".format(
                last_info, start_epoch
            )
        )
    else:
        logger.log("=> do not find the last-info file : {:}".format(last_info))
        start_epoch, valid_accuracies, genotypes = 0, {"best": -1}, {}

    best_arch, best_ntk_score, best_lr_score = search_find_best(train_loader, network, lrc_model, xargs.select_num)
    search_time.update(time.time() - start_time)
    logger.log(
        "RANDOM-NAS finds the best one : {:} with accuracy={:.2f}%, with {:.1f} s.".format(
            best_arch, best_acc, search_time.sum
        )
    )
    if api is not None:
        logger.log("{:}".format(api.query_by_arch(best_arch, "200")))
    logger.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Random search for NAS.")
    parser.add_argument("--data_path", type=str, default='../../cifar.python', help="The path to dataset")
    parser.add_argument("--dataset", type=str, default='cifar10',choices=["cifar10", "cifar100", "ImageNet16-120"], help="Choose between Cifar10/100 and ImageNet-16.")

    # channels and number-of-cells
    parser.add_argument("--search_space_name", type=str, default='nas-bench-201', help="The search space name.")
    parser.add_argument("--config_path", type=str, default='../../configs/nas-benchmark/algos/RANDOM.config', help="The path to the configuration.")
    parser.add_argument("--max_nodes", type=int, default=4, help="The maximum number of nodes.")
    # parser.add_argument("--channel", type=int, default=16, help="The number of channels.")
    # parser.add_argument("--num_cells", type=int, default=5, help="The number of cells in one stage.")
    parser.add_argument("--channel", type=int, default=3, help="The number of channels.")
    parser.add_argument("--num_cells", type=int, default=1, help="The number of cells in one stage.")
    parser.add_argument("--select_num", type=int, default=100, help="The number of selected architectures to evaluate.")
    parser.add_argument("--track_running_stats", type=int, default=0, choices=[0, 1], help="Whether use track_running_stats or not in the BN layer.")
    # log
    parser.add_argument("--workers", type=int, default=4, help="number of data loading workers")
    parser.add_argument("--save_dir", type=str, default='./results/tmp', help="Folder to save checkpoints and log.")
    parser.add_argument("--arch_nas_dataset", type=str, default='../../NAS-Bench-201-v1_1-096897.pth', help="The path to load the architecture dataset (tiny-nas-benchmark).")
    parser.add_argument("--print_freq", type=int, default=200, help="print frequency (default: 200)")
    parser.add_argument("--rand_seed", type=int, default=None, help="manual seed")
    
    args = parser.parse_args(args=[])
    if args.rand_seed is None or args.rand_seed < 0:
        args.rand_seed = random.randint(1, 100000)
    print(args)

    assert torch.cuda.is_available(), "CUDA is not available."
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.set_num_threads(args.workers)
    prepare_seed(args.rand_seed)
    logger = prepare_logger(args)
    
    main(args)