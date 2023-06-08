#!/bin/bash
# python3 -m torch.distributed.launch --nproc_per_node=8 --use_env train.py --data-path '/dataset/ILSVRC2012' --gp --change_qk --relative_position \
# --mode retrain --model_type 'AUTOFORMER' --dist-eval --cfg './experiments/ETF/TF_TAS-T.yaml' --output_dir './OUTPUT/ETF/TF_TAS-T'


#### 4way, 10000iter, seed123
# python3 -m torch.distributed.launch --nproc_per_node=4 --use_env train.py --data-path '/dataset/ILSVRC2012' --gp --change_qk --relative_position \
#  --batch-size 256 --mode retrain --model_type 'AUTOFORMER' --dist-eval --cfg './experiments/ETF-pop10000-seed123/TF_TAS-T.yaml' --output_dir './OUTPUT/ETF-pop10000-seed123/TF_TAS-T'

#  python3 -m torch.distributed.launch --nproc_per_node=4 --use_env train.py --data-path '/dataset/ILSVRC2012' --gp --change_qk --relative_position \
#  --batch-size 64 --mode retrain --model_type 'AUTOFORMER' --dist-eval --cfg './experiments/ETF-pop10000-seed123/TF_TAS-B.yaml' --output_dir './OUTPUT/ETF-pop10000-seed123/TF_TAS-B'

# python3 -m torch.distributed.launch --nproc_per_node=4 --use_env train.py --data-path '/dataset/ILSVRC2012' --gp --change_qk --relative_position \
#  --batch-size 64 --mode retrain --model_type 'AUTOFORMER' --dist-eval --cfg './experiments/ETF-pop10000-seed123/TF_TAS-S.yaml' --output_dir './OUTPUT/ETF-pop10000-seed123/TF_TAS-S'


#### 4way, 8000iter, seed0
# python3 -m torch.distributed.launch --nproc_per_node=4 --use_env train.py --data-path '/dataset/ILSVRC2012' --gp --change_qk --relative_position \
#  --batch-size 64 --mode retrain --model_type 'AUTOFORMER' --dist-eval --cfg './experiments/ETF-pop8000-seed0/TF_TAS-T.yaml' --output_dir './OUTPUT/ETF-pop8000-seed0/TF_TAS-T'

# python3 -m torch.distributed.launch --nproc_per_node=4 --use_env train.py --data-path '/dataset/ILSVRC2012' --gp --change_qk --relative_position \
#  --batch-size 64 --mode retrain --model_type 'AUTOFORMER' --dist-eval --cfg './experiments/ETF-pop8000-seed0/TF_TAS-S.yaml' --output_dir './OUTPUT/ETF-pop8000-seed0/TF_TAS-S'

# python3 -m torch.distributed.launch --nproc_per_node=4 --use_env train.py --data-path '/dataset/ILSVRC2012' --gp --change_qk --relative_position \
#  --batch-size 64 --mode retrain --model_type 'AUTOFORMER' --dist-eval --cfg './experiments/ETF-pop8000-seed0/TF_TAS-B.yaml' --output_dir './OUTPUT/ETF-pop8000-seed0/TF_TAS-B'


#### 8way
# python3 -m torch.distributed.launch --nproc_per_node=8 --use_env train.py --data-path '/dataset/ILSVRC2012' --gp --change_qk --relative_position \
#  --batch-size 256 --mode retrain --model_type 'AUTOFORMER' --dist-eval --cfg './experiments/ETF-pop8000-seed0/TF_TAS-T.yaml' --output_dir './OUTPUT/ETF-pop8000-seed0-bs256x8/TF_TAS-T'

# python3 -m torch.distributed.launch --nproc_per_node=8 --use_env train.py --data-path '/dataset/ILSVRC2012' --gp --change_qk --relative_position \
#  --batch-size 128 --mode retrain --model_type 'AUTOFORMER' --dist-eval --cfg './experiments/ETF-pop8000-seed0/TF_TAS-B.yaml' --output_dir './OUTPUT/ETF-pop8000-seed0-bs128x8/TF_TAS-B'


####
#  python3 -m torch.distributed.launch --nproc_per_node=4 --use_env train.py --data-path '/dataset/ILSVRC2012' --gp --change_qk --relative_position \
#  --batch-size 64 --mode retrain --model_type 'AUTOFORMER' --dist-eval --cfg './experiments/ETF-pop10000-seed123/TF_TAS-B.yaml' --output_dir './OUTPUT/ETF-pop10000-seed123/TF_TAS-B' --resume './OUTPUT/ETF-pop10000-seed123/TF_TAS-B/checkpoint.pth'

python3 -m torch.distributed.launch --nproc_per_node=8 --use_env train.py --data-path '/dataset/ILSVRC2012' --gp --change_qkv --relative_position \
 --batch-size 128 --mode retrain --model_type 'AUTOFORMER' --dist-eval --cfg './experiments/ETF-pop8000-seed0-rev/TF_TAS-S.yaml' --output_dir './OUTPUT/ETF-pop8000-seed0-bs128x8/TF_TAS-S'

python3 -m torch.distributed.launch --nproc_per_node=8 --use_env train.py --data-path '/dataset/ILSVRC2012' --gp --change_qkv --relative_position \
 --batch-size 128 --mode retrain --model_type 'AUTOFORMER' --dist-eval --cfg './experiments/ETF-pop8000-seed0-rev/TF_TAS-B.yaml' --output_dir './OUTPUT/ETF-pop8000-seed0-bs128x8/TF_TAS-B'
