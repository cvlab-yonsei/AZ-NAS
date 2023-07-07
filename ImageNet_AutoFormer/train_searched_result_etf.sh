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


# ##### 4way run
# CUDA_VISIBLE_DEVICES=0,1,2,3, \
# python3 -m torch.distributed.launch --nproc_per_node=4 --use_env train.py --data-path '/dataset/ILSVRC2012' --gp --change_qkv --relative_position \
#  --batch-size 256 --mode retrain --model_type 'AUTOFORMER' --dist-eval --cfg './experiments/ETF-pop8000-seed0-rev/TF_TAS-S.yaml' --init 'xavier_uniform' --output_dir './OUTPUT/ETF-pop8000-seed0-bs128x8-rev-xavier/TF_TAS-S'

# CUDA_VISIBLE_DEVICES=4,5,6,7, \
# python3 -m torch.distributed.launch --master_port 7777 --nproc_per_node=4 --use_env train.py --data-path '/dataset/ILSVRC2012' --gp --change_qkv --relative_position \
#  --batch-size 256 --mode retrain --model_type 'AUTOFORMER' --dist-eval --cfg './experiments/ETF-pop8000-seed0-rev/TF_TAS-S.yaml' --init 'trunc_normal' --output_dir './OUTPUT/ETF-pop8000-seed0-bs128x8-rev-trunc/TF_TAS-S'

# python3 -m torch.distributed.launch --nproc_per_node=8 --use_env train.py --data-path '/dataset/ILSVRC2012' --gp --change_qkv --relative_position \
#  --batch-size 128 --mode retrain --model_type 'AUTOFORMER' --dist-eval --cfg './experiments/ETF-pop8000-seed0-rev/TF_TAS-B.yaml' --output_dir './OUTPUT/ETF-pop8000-seed0-bs128x8-rev/TF_TAS-B'

# CUDA_VISIBLE_DEVICES=0,1,2,3, \
# python3 -m torch.distributed.launch --nproc_per_node=4 --use_env train.py --lr 2.5e-4 --data-path '/dataset/ILSVRC2012' --gp --change_qkv --relative_position \
#  --batch-size 256 --mode retrain --model_type 'AUTOFORMER' --dist-eval --cfg './experiments/ETF-pop8000-seed0-rev/TF_TAS-S.yaml' --init 'xavier_uniform' --output_dir './OUTPUT/ETF-pop8000-seed0-bs128x8-rev-xavier/TF_TAS-S-2.5e-4'

######
 CUDA_VISIBLE_DEVICES=0,2,4,6, \
python3 -m torch.distributed.launch --nproc_per_node=4 --use_env train.py --data-path '/dataset/ILSVRC2012' --gp --change_qkv --relative_position \
 --batch-size 256 --mode retrain --model_type 'AUTOFORMER' --dist-eval --cfg './experiments/ETF-pop8000-seed0-cap-trunc/Small.yaml' --init 'trunc_normal' --output_dir './OUTPUT/ETF-pop8000-seed0-bs256x4-cap-trunc/Small'

 CUDA_VISIBLE_DEVICES=1,3,5,7, \
python3 -m torch.distributed.launch --master_port 7777 --nproc_per_node=4 --use_env train.py --data-path '/dataset/ILSVRC2012' --gp --change_qkv --relative_position \
 --batch-size 256 --mode retrain --model_type 'AUTOFORMER' --dist-eval --cfg './experiments/ETF-pop8000-seed0-cap-trunc/Tiny.yaml' --init 'trunc_normal' --output_dir './OUTPUT/ETF-pop8000-seed0-bs256x4-cap-trunc/Tiny'

 CUDA_VISIBLE_DEVICES=1,3,5,7, \
python3 -m torch.distributed.launch --master_port 7777 --nproc_per_node=4 --use_env train.py --data-path '/dataset/ILSVRC2012' --gp --change_qkv --relative_position \
 --batch-size 128 --mode retrain --model_type 'AUTOFORMER' --dist-eval --cfg './experiments/ETF-pop8000-seed0-cap-trunc/Base.yaml' --init 'trunc_normal' --output_dir './OUTPUT/ETF-pop8000-seed0-bs128x4-cap-trunc/Base'

 ####
 python3 -m torch.distributed.launch --nproc_per_node=8 --use_env train.py --data-path '/dataset/ILSVRC2012' --gp --change_qkv --relative_position \
 --batch-size 64 --mode retrain --model_type 'AUTOFORMER' --dist-eval --cfg './experiments/ETF-pop10000-seed123-cap-trunc/Base.yaml' --init 'trunc_normal' --output_dir './OUTPUT/ETF-pop10000-seed123-cap-trunc/Base-bs64x8' --resume './OUTPUT/ETF-pop10000-seed123-cap-trunc/Base-bs64x8/checkpoint.pth'

 ### 4way bs 128*4
 python3 -m torch.distributed.launch --nproc_per_node=4 --use_env train.py --data-path '/dataset/ILSVRC2012' --gp --change_qkv --relative_position \
 --batch-size 128 --mode retrain --model_type 'AUTOFORMER' --dist-eval --cfg './experiments/ETF-pop10000-seed123-cap-trunc/Tiny.yaml' --init 'trunc_normal' --output_dir './OUTPUT/ETF-pop10000-seed123-cap-trunc/Tiny-bs128x4'

 python3 -m torch.distributed.launch --nproc_per_node=4 --use_env train.py --data-path '/dataset/ILSVRC2012' --gp --change_qkv --relative_position \
 --batch-size 128 --mode retrain --model_type 'AUTOFORMER' --dist-eval --cfg './experiments/ETF-pop10000-seed123-cap-trunc/Small.yaml' --init 'trunc_normal' --output_dir './OUTPUT/ETF-pop10000-seed123-cap-trunc/Small-bs128x4'

 ### debug
 python3 -m torch.distributed.launch --nproc_per_node=8 --use_env train.py --data-path '/dataset/ILSVRC2012' --gp --change_qkv --relative_position \
 --batch-size 64 --mode retrain --model_type 'AUTOFORMER' --dist-eval --cfg './experiments/ETF-pop10000-seed123-cap-trunc/Base.yaml' --init 'trunc_normal' --output_dir './OUTPUT/ETF-pop10000-seed123-cap-trunc/Base-bs64x8' --resume './OUTPUT/ETF-pop10000-seed123-cap-trunc/Base-bs64x8/checkpoint.pth'

torchrun --nproc_per_node=8 train.py --data-path '/dataset/ILSVRC2012' --gp --change_qkv --relative_position \
 --batch-size 64 --mode retrain --model_type 'AUTOFORMER' --dist-eval --cfg './experiments/ETF-pop10000-seed123-cap-trunc/Base.yaml' --init 'trunc_normal' --output_dir './OUTPUT/ETF-pop10000-seed123-cap-trunc/Base-bs64x8-2nd'

#  CUDA_VISIBLE_DEVICES=0,2,4,6, OMP_NUM_THREADS=8 \
#  torchrun --nproc_per_node=4 train.py --data-path '/dataset/ILSVRC2012' --gp --change_qkv --relative_position \
#  --batch-size 128 --mode retrain --model_type 'AUTOFORMER' --dist-eval --cfg './experiments/ETF-pop10000-seed123-cap-trunc/Base.yaml' --init 'trunc_normal' --output_dir './OUTPUT/ETF-pop10000-seed123-cap-trunc/Base-bs64x8' --resume './OUTPUT/ETF-pop10000-seed123-cap-trunc/Base-bs64x8/checkpoint.pth'

# CUDA_VISIBLE_DEVICES=1,3,5,7, OMP_NUM_THREADS=8 \
#  torchrun --master_port 7777 --nproc_per_node=4 train.py --data-path '/dataset/ILSVRC2012' --gp --change_qkv --relative_position \
#  --batch-size 128 --mode retrain --model_type 'AUTOFORMER' --dist-eval --cfg './experiments/ETF-pop10000-seed123-cap-trunc/Base.yaml' --init 'trunc_normal' --output_dir './OUTPUT/ETF-pop10000-seed123-cap-trunc/Base-bs128x4'

### 4way bs 256*4
 python3 -m torch.distributed.launch --nproc_per_node=4 --use_env train_half_bs.py --data-path '/dataset/ILSVRC2012' --gp --change_qkv --relative_position \
 --batch-size 256 --mode retrain --model_type 'AUTOFORMER' --dist-eval --cfg './experiments/ETF-pop10000-seed123-cap-trunc/Tiny.yaml' --init 'trunc_normal' --output_dir './OUTPUT/ETF-pop10000-seed123-cap-trunc/Tiny-bs256x4'

 python3 -m torch.distributed.launch --nproc_per_node=4 --use_env train_half_bs.py --data-path '/dataset/ILSVRC2012' --gp --change_qkv --relative_position \
 --batch-size 256 --mode retrain --model_type 'AUTOFORMER' --dist-eval --cfg './experiments/ETF-pop10000-seed123-cap-trunc/Small.yaml' --init 'trunc_normal' --output_dir './OUTPUT/ETF-pop10000-seed123-cap-trunc/Small-bs256x4'