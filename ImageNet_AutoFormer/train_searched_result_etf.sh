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
 --batch-size 256 --mode retrain --model_type 'AUTOFORMER' --dist-eval --cfg './experiments/ETF-pop10000-seed123-cap-trunc/Tiny.yaml' --output_dir './OUTPUT/ETF-pop10000-seed123-cap-trunc/Tiny-bs256x4' --no-amp

 python3 -m torch.distributed.launch --nproc_per_node=4 --use_env train_half_bs.py --data-path '/dataset/ILSVRC2012' --gp --change_qkv --relative_position \
 --batch-size 256 --mode retrain --model_type 'AUTOFORMER' --dist-eval --cfg './experiments/ETF-pop10000-seed123-cap-trunc/Small.yaml' --output_dir './OUTPUT/ETF-pop10000-seed123-cap-trunc/Small-bs256x4' --no-amp

### using subnet / 4way base bs 64*4 
python3 -m torch.distributed.launch --nproc_per_node=4 --use_env train_subnet.py --data-path '/dataset/ILSVRC2012' --gp --change_qkv --relative_position \
 --batch-size 64 --mode retrain --model_type 'AUTOFORMER' --dist-eval --cfg './experiments/ETF-pop10000-seed123-cap-trunc/Base.yaml' --output_dir './OUTPUT/ETF-pop10000-seed123-cap-trunc/Base-bs64x4'

### using subnet / 4way base bs 128*4 
CUDA_VISIBLE_DEVICES=0,1,2,3, \
python3 -m torch.distributed.launch --nproc_per_node=4 --use_env train_subnet.py --data-path '/dataset/ILSVRC2012' --gp --change_qkv --relative_position \
 --batch-size 128 --mode retrain --model_type 'AUTOFORMER' --dist-eval --cfg './experiments/ETF-pop10000-seed123-cap-trunc/Base.yaml' --output_dir './OUTPUT/ETF-pop10000-seed123-cap-trunc/Base-bs128x4-subnet'

### using subnet / 4way small bs 256*4 
CUDA_VISIBLE_DEVICES=4,5,6,7, \
python3 -m torch.distributed.launch --master_port 7777 --nproc_per_node=4 --use_env train_subnet.py --data-path '/dataset/ILSVRC2012' --gp --change_qkv --relative_position \
 --batch-size 256 --mode retrain --model_type 'AUTOFORMER' --dist-eval --cfg './experiments/ETF-pop10000-seed123-cap-trunc/Small.yaml' --output_dir './OUTPUT/ETF-pop10000-seed123-cap-trunc/Small-bs256x4-subnet'

### using subnet / 4way tiny bs 256*4 
python3 -m torch.distributed.launch --nproc_per_node=4 --use_env train_subnet.py --data-path '/dataset/ILSVRC2012' --gp --change_qkv --relative_position \
 --batch-size 256 --mode retrain --model_type 'AUTOFORMER' --dist-eval --cfg './experiments/ETF-pop10000-seed123-cap-trunc/Tiny.yaml' --output_dir './OUTPUT/ETF-pop10000-seed123-cap-trunc/Tiny-bs256x4-subnet'

### using subnet / 4way tiny nocap bs 256*4 
python3 -m torch.distributed.launch --nproc_per_node=4 --use_env train_subnet.py --data-path '/dataset/ILSVRC2012' --gp --change_qkv --relative_position \
 --batch-size 256 --mode retrain --model_type 'AUTOFORMER' --dist-eval --cfg './experiments/ETF-pop10000-seed0-nocap-trunc/Tiny.yaml' --output_dir './OUTPUT/ETF-pop10000-seed0-nocap-trunc/Tiny-bs256x4-subnet'

CUDA_VISIBLE_DEVICES=0,1,2,3, \
python3 -m torch.distributed.launch --nproc_per_node=4 --use_env train_subnet.py --data-path '/dataset/ILSVRC2012' --gp --change_qkv --relative_position \
 --batch-size 128 --mode retrain --model_type 'AUTOFORMER' --dist-eval --cfg './experiments/ETF-pop10000-seed0-nocap-trunc/Base.yaml' --output_dir './OUTPUT/ETF-pop10000-seed0-nocap-trunc/Base-bs128x4-subnet'

CUDA_VISIBLE_DEVICES=4,5,6,7, \
python3 -m torch.distributed.launch --master_port 7777 --nproc_per_node=4 --use_env train_subnet.py --data-path '/dataset/ILSVRC2012' --gp --change_qkv --relative_position \
 --batch-size 256 --mode retrain --model_type 'AUTOFORMER' --dist-eval --cfg './experiments/ETF-pop10000-seed0-nocap-trunc/Small.yaml' --output_dir './OUTPUT/ETF-pop10000-seed0-nocap-trunc/Small-bs256x4-subnet'

###### 8000 cap
 CUDA_VISIBLE_DEVICES=6,7, \
python3 -m torch.distributed.launch --nproc_per_node=2 --use_env train_subnet.py --data-path '/dataset/ILSVRC2012' --gp --change_qkv --relative_position \
 --batch-size 512 --mode retrain --model_type 'AUTOFORMER' --dist-eval --cfg './experiments/ETF-pop8000-seed0-cap-trunc/Small.yaml' --output_dir './OUTPUT/ETF-pop8000-seed0-bs512x2-cap-trunc/Small' --resume './OUTPUT/ETF-pop8000-seed0-bs512x2-cap-trunc/Small/checkpoint.pth'

 CUDA_VISIBLE_DEVICES=4,5, \
python3 -m torch.distributed.launch --master_port 8888 --nproc_per_node=2 --use_env train_subnet.py --data-path '/dataset/ILSVRC2012' --gp --change_qkv --relative_position \
 --batch-size 512 --mode retrain --model_type 'AUTOFORMER' --dist-eval --cfg './experiments/ETF-pop8000-seed0-cap-trunc/Tiny.yaml' --output_dir './OUTPUT/ETF-pop8000-seed0-bs512x2-cap-trunc/Tiny'  --resume './OUTPUT/ETF-pop8000-seed0-bs512x2-cap-trunc/Tiny/checkpoint.pth'

 CUDA_VISIBLE_DEVICES=0,1,2,3, \
python3 -m torch.distributed.launch --master_port 7777 --nproc_per_node=4 --use_env train_subnet.py --data-path '/dataset/ILSVRC2012' --gp --change_qkv --relative_position \
 --batch-size 128 --mode retrain --model_type 'AUTOFORMER' --dist-eval --cfg './experiments/ETF-pop8000-seed0-cap-trunc/Base.yaml' --output_dir './OUTPUT/ETF-pop8000-seed0-bs128x4-cap-trunc/Base' --resume './OUTPUT/ETF-pop8000-seed0-bs128x4-cap-trunc/Base/checkpoint.pth'

CUDA_VISIBLE_DEVICES=0,1,4,5, \
python3 -m torch.distributed.launch --master_port 7777 --nproc_per_node=4 --use_env train_subnet.py --data-path '/dataset/ILSVRC2012' --gp --change_qkv --relative_position \
 --batch-size 128 --mode retrain --model_type 'AUTOFORMER' --dist-eval --cfg './experiments/ETF-pop8000-seed0-cap-trunc/Base.yaml' --output_dir './OUTPUT/ETF-pop8000-seed0-cap-trunc/Base-bs128x4-subnet_no_sample' --resume './OUTPUT/ETF-pop8000-seed0-cap-trunc/Base-bs128x4-subnet_no_sample/checkpoint.pth'

CUDA_VISIBLE_DEVICES=6,7, CUDA_LAUNCH_BLOCKING=1 \
python3 -m torch.distributed.launch --master_port 6666 --nproc_per_node=2 --use_env train_subnet.py --data-path '/dataset/ILSVRC2012' --gp --change_qkv --relative_position \
 --batch-size 512 --mode retrain --model_type 'AUTOFORMER' --dist-eval --cfg './experiments/ETF-pop8000-seed0-cap-trunc/Small.yaml' --output_dir './OUTPUT/ETF-pop8000-seed0-cap-trunc/Small-bs512x2-subnet_no_sample' --resume './OUTPUT/ETF-pop8000-seed0-cap-trunc/Small-bs512x2-subnet_no_sample/checkpoint.pth'

 ###### Expressivity + Trainability + Complexity / seed 123 / iter 10000
CUDA_VISIBLE_DEVICES=0,1,4,5, CUDA_LAUNCH_BLOCKING=1 \
python3 -m torch.distributed.launch --master_port 7777 --nproc_per_node=4 --use_env train_subnet.py --data-path '/dataset/ILSVRC2012' --gp --change_qkv --relative_position \
 --epochs 500 --warmup-epochs 20 --batch-size 256 --mode retrain --model_type 'AUTOFORMER' --dist-eval --cfg './experiments/ETF-ETC-seed123-iter10000/Base.yaml' --output_dir './OUTPUT/ETF-ETC-seed123-iter10000/Base-bs256x4-use_subnet-500ep' --resume './OUTPUT/ETF-ETC-seed123-iter10000/Base-bs256x4-use_subnet-500ep/checkpoint.pth'

 CUDA_VISIBLE_DEVICES=6,7, CUDA_LAUNCH_BLOCKING=1 \
python3 -m torch.distributed.launch --master_port 6666 --nproc_per_node=2 --use_env train_subnet.py --data-path '/dataset/ILSVRC2012' --gp --change_qkv --relative_position \
 --epochs 500 --warmup-epochs 20 --batch-size 512 --mode retrain --model_type 'AUTOFORMER' --dist-eval --cfg './experiments/ETF-ETC-seed123-iter10000/Small.yaml' --output_dir './OUTPUT/ETF-ETC-seed123-iter10000/Small-bs512x2-use_subnet-500ep' --resume './OUTPUT/ETF-ETC-seed123-iter10000/Small-bs512x2-use_subnet-500ep/checkpoint.pth'

CUDA_VISIBLE_DEVICES=0,2,4,6, CUDA_LAUNCH_BLOCKING=1 \
python3 -m torch.distributed.launch --master_port 8888 --nproc_per_node=4 --use_env train_subnet.py --data-path '/dataset/ILSVRC2012' --gp --change_qkv --relative_position \
 --epochs 500 --warmup-epochs 20 --batch-size 256 --mode retrain --model_type 'AUTOFORMER' --dist-eval --cfg './experiments/ETF-ETC-seed123-iter10000/Tiny.yaml' --output_dir './OUTPUT/ETF-ETC-seed123-iter10000/Tiny-bs256x4-use_subnet-500ep'

CUDA_LAUNCH_BLOCKING=1 \
python3 -m torch.distributed.launch --master_port 7777 --nproc_per_node=8 --use_env train_subnet.py --data-path '/dataset/ILSVRC2012' --gp --change_qkv --relative_position \
 --epochs 500 --warmup-epochs 20 --batch-size 128 --mode retrain --model_type 'AUTOFORMER' --dist-eval --cfg './experiments/ETF-ETC-seed123-iter10000/Base.yaml' --output_dir './OUTPUT/ETF-ETC-seed123-iter10000/Base-bs128x8-use_subnet-500ep' --resume './OUTPUT/ETF-ETC-seed123-iter10000/Base-bs128x8-use_subnet-500ep/checkpoint.pth'

CUDA_LAUNCH_BLOCKING=1 \
python3 -m torch.distributed.launch --master_port 7777 --nproc_per_node=8 --use_env train_subnet.py --data-path '/dataset/ILSVRC2012' --gp --change_qkv --relative_position \
 --epochs 500 --warmup-epochs 20 --batch-size 128 --mode retrain --model_type 'AUTOFORMER' --dist-eval --cfg './experiments/ETF-ETC-seed123-iter10000/Base.yaml' --output_dir './OUTPUT/ETF-ETC-seed123-iter10000/Base-bs128x8-use_subnet-500ep-no_amp' --no-amp # setting amp=True occasionally results in a loss value of NaN for the Base model

CUDA_VISIBLE_DEVICES=0,1,2,3, CUDA_LAUNCH_BLOCKING=1 \
python3 -m torch.distributed.launch --master_port 7777 --nproc_per_node=4 --use_env train_subnet.py --data-path '/dataset/ILSVRC2012' --gp --change_qkv --relative_position \
 --epochs 300 --warmup-epochs 20 --batch-size 256 --mode retrain --model_type 'AUTOFORMER' --dist-eval --cfg './experiments/ETF-ETC-seed123-iter10000/Base.yaml' --output_dir './OUTPUT/ETF-ETC-seed123-iter10000/Base-bs256x4-use_subnet-300ep' --resume './OUTPUT/ETF-ETC-seed123-iter10000/Base-bs256x4-use_subnet-300ep/checkpoint.pth'

CUDA_LAUNCH_BLOCKING=1 \
python3 -m torch.distributed.launch --master_port 7777 --nproc_per_node=8 --use_env train_subnet.py --data-path '/dataset/ILSVRC2012' --gp --change_qkv --relative_position \
 --epochs 300 --warmup-epochs 20 --batch-size 128 --mode retrain --model_type 'AUTOFORMER' --dist-eval --cfg './experiments/ETF-ETC-seed123-iter10000/Base.yaml' --output_dir './OUTPUT/ETF-ETC-seed123-iter10000/Base-bs128x8-use_subnet-300ep-rev_amp'

###### Expressivity + Trainability + Complexity / seed 123 / iter 10000 / TFTAS budget
CUDA_VISIBLE_DEVICES=6,7, CUDA_LAUNCH_BLOCKING=1 \
python3 -m torch.distributed.launch --master_port 6666 --nproc_per_node=2 --use_env train_subnet.py --data-path '/dataset/ILSVRC2012' --gp --change_qkv --relative_position \
 --epochs 500 --warmup-epochs 20 --batch-size 512 --mode retrain --model_type 'AUTOFORMER' --dist-eval --cfg './experiments/ETF-ETC-seed123-iter10000-TFTAS_budget/Tiny.yaml' --output_dir './OUTPUT/ETF-ETC-seed123-iter10000-TFTAS_budget/Tiny-bs512x2-use_subnet-500ep'

CUDA_VISIBLE_DEVICES=4,5,6,7, CUDA_LAUNCH_BLOCKING=1 \
python3 -m torch.distributed.launch --master_port 6666 --nproc_per_node=4 --use_env train_subnet.py --data-path '/dataset/ILSVRC2012' --gp --change_qkv --relative_position \
 --epochs 500 --warmup-epochs 20 --batch-size 256 --mode retrain --model_type 'AUTOFORMER' --dist-eval --cfg './experiments/ETF-ETC-seed123-iter10000-TFTAS_budget/Small.yaml' --output_dir './OUTPUT/ETF-ETC-seed123-iter10000-TFTAS_budget/Small-bs256x4-use_subnet-500ep'

CUDA_LAUNCH_BLOCKING=1 \
python3 -m torch.distributed.launch --master_port 7777 --nproc_per_node=8 --use_env train_subnet.py --data-path '/dataset/ILSVRC2012' --gp --change_qkv --relative_position \
 --epochs 300 --warmup-epochs 20 --batch-size 128 --mode retrain --model_type 'AUTOFORMER' --dist-eval --cfg './experiments/ETF-ETC-seed123-iter10000-TFTAS_budget/Base.yaml' --output_dir './OUTPUT/ETF-ETC-seed123-iter10000-TFTAS_budget/Base-bs128x8-use_subnet-300ep'