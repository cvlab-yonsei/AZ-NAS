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
from xautodl.models import get_search_spaces
from custom_models import get_cell_based_tiny_net

# NB201
from nas_201_api import NASBench201API as API

from search_valid_train import *


def main(xargs):
    train_data, valid_data, xshape, class_num = get_datasets(xargs.dataset, xargs.data_path, -1)
    config = load_config(xargs.config_path, {"class_num": class_num, "xshape": xshape}, logger)
    search_loader, _, valid_loader = get_nas_search_loaders(train_data,
                                                            valid_data,
                                                            xargs.dataset,
                                                            "./configs/nas-benchmark",
                                                            (config.batch_size, config.test_batch_size),
                                                            xargs.workers)
    logger.log("||||||| {:10s} ||||||| Search-Loader-Num={:}, Valid-Loader-Num={:}, batch size={:}".format(
                xargs.dataset, len(search_loader), len(valid_loader), config.batch_size))
    logger.log("||||||| {:10s} ||||||| Config={:}".format(xargs.dataset, config))

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

    w_optimizer, w_scheduler, criterion = get_optim_scheduler(search_model.parameters(), config)

    logger.log("w-optimizer : {:}".format(w_optimizer))
    logger.log("w-scheduler : {:}".format(w_scheduler))
    logger.log("criterion   : {:}".format(criterion))
    # if xargs.arch_nas_dataset is None:
    #     api = None
    # else:
    #     api = API(xargs.arch_nas_dataset)
    api = None
    logger.log("{:} create API = {:} done".format(time_string(), api))

    last_info, model_base_path, model_best_path = (
        logger.path("info"),
        logger.path("model"),
        logger.path("best"),
    )
    network, criterion = torch.nn.DataParallel(search_model).cuda(), criterion.cuda()

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
        w_scheduler.load_state_dict(checkpoint["w_scheduler"])
        w_optimizer.load_state_dict(checkpoint["w_optimizer"])
        logger.log(
            "=> loading checkpoint of the last-info '{:}' start with {:}-th epoch.".format(
                last_info, start_epoch
            )
        )
    else:
        logger.log("=> do not find the last-info file : {:}".format(last_info))
        start_epoch, valid_accuracies, genotypes = 0, {"best": -1}, {}

    # start training
    start_time, search_time, epoch_time, total_epoch = (
        time.time(),
        AverageMeter(),
        AverageMeter(),
        config.epochs + config.warmup,
    )
    for epoch in range(start_epoch, total_epoch):
        w_scheduler.update(epoch, 0.0)
        need_time = "Time Left: {:}".format(
            convert_secs2time(epoch_time.val * (total_epoch - epoch), True)
        )
        epoch_str = "{:03d}-{:03d}".format(epoch, total_epoch)
        logger.log(
            "\n[Search the {:}-th epoch] {:}, LR={:}".format(
                epoch_str, need_time, min(w_scheduler.get_lr())
            )
        )

        # selected_arch = search_find_best(valid_loader, network, criterion, xargs.select_num)
        search_w_loss, search_w_top1, search_w_top5 = search_func(
            search_loader,
            network,
            criterion,
            w_scheduler,
            w_optimizer,
            epoch_str,
            xargs.print_freq,
            logger,
        )
        search_time.update(time.time() - start_time)
        logger.log(
            "[{:}] searching : loss={:.2f}, accuracy@1={:.2f}%, accuracy@5={:.2f}%, time-cost={:.1f} s".format(
                epoch_str, search_w_loss, search_w_top1, search_w_top5, search_time.sum
            )
        )
        valid_a_loss, valid_a_top1, valid_a_top5 = valid_func(
            valid_loader, network, criterion
        )
        logger.log(
            "[{:}] evaluate  : loss={:.2f}, accuracy@1={:.2f}%, accuracy@5={:.2f}%".format(
                epoch_str, valid_a_loss, valid_a_top1, valid_a_top5
            )
        )
        cur_arch, cur_valid_acc = search_find_best(
            valid_loader, network, xargs.select_num
        )
        logger.log(
            "[{:}] find-the-best : {:}, accuracy@1={:.2f}%".format(
                epoch_str, cur_arch, cur_valid_acc
            )
        )
        genotypes[epoch] = cur_arch
        # check the best accuracy
        valid_accuracies[epoch] = valid_a_top1
        if valid_a_top1 > valid_accuracies["best"]:
            valid_accuracies["best"] = valid_a_top1
            find_best = True
        else:
            find_best = False

        # save checkpoint
        save_path = save_checkpoint(
            {
                "epoch": epoch + 1,
                "args": deepcopy(xargs),
                "search_model": search_model.state_dict(),
                "w_optimizer": w_optimizer.state_dict(),
                "w_scheduler": w_scheduler.state_dict(),
                "genotypes": genotypes,
                "valid_accuracies": valid_accuracies,
            },
            model_base_path,
            logger,
        )
        last_info = save_checkpoint(
            {
                "epoch": epoch + 1,
                "args": deepcopy(args),
                "last_checkpoint": save_path,
            },
            logger.path("info"),
            logger,
        )
        if find_best:
            logger.log(
                "<<<--->>> The {:}-th epoch : find the highest validation accuracy : {:.2f}%.".format(
                    epoch_str, valid_a_top1
                )
            )
            copy_checkpoint(model_base_path, model_best_path, logger)
        if api is not None:
            logger.log("{:}".format(api.query_by_arch(genotypes[epoch], "200")))
        # measure elapsed time
        epoch_time.update(time.time() - start_time)
        start_time = time.time()

    logger.log("\n" + "-" * 200)
    logger.log("Pre-searching costs {:.1f} s".format(search_time.sum))
    start_time = time.time()
    best_arch, best_acc = search_find_best(valid_loader, network, xargs.select_num)
    search_time.update(time.time() - start_time)
    logger.log(
        "RANDOM-NAS finds the best one : {:} with accuracy={:.2f}%, with {:.1f} s.".format(
            best_arch, best_acc, search_time.sum
        )
    )
    if api is not None:
        logger.log("{:}".format(api.query_by_arch(best_arch, "200")))
    logger.close()

    print("Start training")
    train_best_arch(xargs, network, best_arch)

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Random search for NAS.")
    parser.add_argument("--data_path", type=str, default='./cifar.python', help="The path to dataset")
    parser.add_argument("--dataset", type=str, default='cifar10',choices=["cifar10", "cifar100", "ImageNet16-120"], help="Choose between Cifar10/100 and ImageNet-16.")

    # channels and number-of-cells
    parser.add_argument("--search_space_name", type=str, default='nas-bench-201', help="The search space name.")
    parser.add_argument("--config_path", type=str, default='./configs/nas-benchmark/algos/RANDOM.config', help="The path to the configuration.")
    parser.add_argument("--train_config_path", type=str, default='./configs/nas-benchmark/CIFAR.config', help="The path to the configuration.")
    parser.add_argument("--max_nodes", type=int, default=4, help="The maximum number of nodes.")
    parser.add_argument("--channel", type=int, default=16, help="The number of channels.")
    parser.add_argument("--num_cells", type=int, default=5, help="The number of cells in one stage.")
    parser.add_argument("--select_num", type=int, default=100, help="The number of selected architectures to evaluate.")
    parser.add_argument("--track_running_stats", type=int, default=0, choices=[0, 1], help="Whether use track_running_stats or not in the BN layer.")
    # log
    parser.add_argument("--workers", type=int, default=4, help="number of data loading workers")
    parser.add_argument("--save_dir", type=str, default='./results/NB201/RSPS-CIFAR10-BN0-cell_level', help="Folder to save checkpoints and log.")
    # parser.add_argument("--arch_nas_dataset", type=str, default='./NAS-Bench-201-v1_1-096897.pth', help="The path to load the architecture dataset (tiny-nas-benchmark).")
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