'''
Copyright (C) 2010-2021 Alibaba Group Holding Limited.
'''


import os, sys
this_script_dir = os.path.dirname(os.path.abspath(__file__))

import global_utils
import torch
import urllib.request
from . import masternet

pretrain_model_pth_dir = os.path.expanduser('~/.cache/pytorch/checkpoints/ZiCo_pretrained')

ZiCo_model_zoo = {
    'ZiCo_imagenet1k_flops450M_res224': {
        'plainnet_str_txt': 'ZiCo_imagenet1k_flops450M_res224.txt',
        'pth_path': 'ZiCo_imagenet1k_flops450M_res224/student_best-params_rank0.pth',
        'num_classes': 1000,
        'use_SE': True,
        'resolution': 224,
        'crop_image_size': 320,
    },

    'ZiCo_imagenet1k_flops600M_res224': {
        'plainnet_str_txt': 'ZiCo_imagenet1k_flops600M_res224.txt',
        'pth_path': 'ZiCo_imagenet1k_flops600M_res224/student_best-params_rank0.pth',
        'num_classes': 1000,
        'use_SE': True,
        'resolution': 224,
        'crop_image_size': 320,
    },

    'ZiCo_imagenet1k_flops1G_res224': {
        'plainnet_str_txt': 'ZiCo_imagenet1k_flops1G_res224.txt',
        'pth_path': 'ZiCo_imagenet1k_flops1G_res224/student_best-params_rank0.pth',
        'num_classes': 1000,
        'use_SE': True,
        'resolution': 224,
        'crop_image_size': 320,
    },

    'Params_imagenet1k_flops450M_res224': {
        'plainnet_str_txt': 'Params_imagenet1k_flops450M_res224.txt',
        'pth_path': 'Params_imagenet1k_flops450M_res224/student_best-params_rank0.pth',
        'num_classes': 1000,
        'use_SE': True,
        'resolution': 224,
        'crop_image_size': 320,
    },
}

def get_ZiCo(model_name, pretrained=False, ckptpath=None):
    if model_name not in ZiCo_model_zoo:
        print('Error! Cannot find ZiCo model name! Please choose one in the following list:')

        for key in ZiCo_model_zoo:
            print(key)
        raise ValueError('ZiCo Model Name not found: ' + model_name)

    model_plainnet_str_txt = os.path.join(this_script_dir, ZiCo_model_zoo[model_name]['plainnet_str_txt'])
    if ckptpath is None:
        model_pth_path = os.path.join(pretrain_model_pth_dir, ZiCo_model_zoo[model_name]['pth_path'])
    else: 
        model_pth_path = ckptpath
    use_SE = ZiCo_model_zoo[model_name]['use_SE']
    num_classes = ZiCo_model_zoo[model_name]['num_classes']

    with open(model_plainnet_str_txt, 'r') as fid:
        model_plainnet_str = fid.readline().strip()

    model = masternet.PlainNet(num_classes=num_classes, plainnet_struct=model_plainnet_str, use_se=use_SE)

    if pretrained:
        print('loading pretrained parameters...')
        checkpoint = torch.load(model_pth_path, map_location='cpu')
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint

        model.load_state_dict(state_dict, strict=True)

    return model