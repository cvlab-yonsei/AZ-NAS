#!/bin/bash
# python3 -m torch.distributed.launch --nproc_per_node=8 --use_env train.py --data-path '/dataset/ILSVRC2012' --gp --change_qk --relative_position \
# --mode retrain --model_type 'AUTOFORMER' --dist-eval --cfg './experiments/ETF/TF_TAS-T.yaml' --output_dir './OUTPUT/ETF/TF_TAS-T'

# python3 -m torch.distributed.launch --nproc_per_node=4 --use_env train.py --data-path '/dataset/ILSVRC2012' --gp --change_qk --relative_position \
#  --batch-size 256 --mode retrain --model_type 'AUTOFORMER' --dist-eval --cfg './experiments/ETF-pop10000-seed123/TF_TAS-T.yaml' --output_dir './OUTPUT/ETF-pop10000-seed123/TF_TAS-T'

 python3 -m torch.distributed.launch --nproc_per_node=4 --use_env train.py --data-path '/dataset/ILSVRC2012' --gp --change_qk --relative_position \
 --batch-size 64 --mode retrain --model_type 'AUTOFORMER' --dist-eval --cfg './experiments/ETF-pop10000-seed123/TF_TAS-B.yaml' --output_dir './OUTPUT/ETF-pop10000-seed123/TF_TAS-B'