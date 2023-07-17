#!/bin/bash
# CUDA_VISIBLE_DEVICES=0, python3 search_autoformer_etf.py --data-path '/dataset/ILSVRC2012' --gp \
#  --param-limits 54 --change_qk --relative_position --dist-eval --cfg './experiments/search_space/space-B.yaml' --output_dir './OUTPUT/ETF-space-B'

# CUDA_VISIBLE_DEVICES=1, python3 search_autoformer_etf.py --data-path '/dataset/ILSVRC2012' --gp \
#  --param-limits 23 --change_qk --relative_position --dist-eval --cfg './experiments/search_space/space-S.yaml' --output_dir './OUTPUT/ETF-space-S'

# CUDA_VISIBLE_DEVICES=1, python3 search_autoformer_etf.py --data-path '/dataset/ILSVRC2012' --gp \
#   --param-limits 6 --change_qk --relative_position --dist-eval --cfg './experiments/search_space/space-T.yaml' --output_dir './OUTPUT/ETF-space-T'


### 10000 archs
# CUDA_VISIBLE_DEVICES=0, python3 search_autoformer_etf.py --data-path '/dataset/ILSVRC2012' --gp --population-num 10000 --seed 123 \
#  --param-limits 54 --min-param-limits 52 --change_qk --relative_position --dist-eval --cfg './experiments/search_space/space-B.yaml' --output_dir './OUTPUT/ETF-space-B-pop10000'

# CUDA_VISIBLE_DEVICES=1, python3 search_autoformer_etf.py --data-path '/dataset/ILSVRC2012' --gp --population-num 10000 --seed 123  \
#   --param-limits 6 --min-param-limits 4 --change_qk --relative_position --dist-eval --cfg './experiments/search_space/space-T.yaml' --output_dir './OUTPUT/ETF-space-T-pop10000'

# CUDA_VISIBLE_DEVICES=1, python3 search_autoformer_etf.py --data-path '/dataset/ILSVRC2012' --gp --population-num 10000 --seed 123  \
#  --param-limits 23 --min-param-limits 21 --change_qk --relative_position --dist-eval --cfg './experiments/search_space/space-S.yaml' --output_dir './OUTPUT/ETF-space-S-pop10000'

# ### 8000 archs
# CUDA_VISIBLE_DEVICES=0, python3 search_autoformer_etf.py --data-path '/dataset/ILSVRC2012' --gp --population-num 8000 \
#  --param-limits 54 --min-param-limits 52 --change_qkv --relative_position --dist-eval --cfg './experiments/search_space/space-B.yaml' --output_dir './OUTPUT/ETF-space-B-pop8000-seed0-rev'

# CUDA_VISIBLE_DEVICES=1, python3 search_autoformer_etf.py --data-path '/dataset/ILSVRC2012' --gp --population-num 8000 \
#   --param-limits 6 --min-param-limits 4 --change_qkv --relative_position --dist-eval --cfg './experiments/search_space/space-T.yaml' --output_dir './OUTPUT/ETF-space-T-pop8000-seed0-rev'

# CUDA_VISIBLE_DEVICES=1, python3 search_autoformer_etf.py --data-path '/dataset/ILSVRC2012' --gp --population-num 8000 \
#  --param-limits 23 --min-param-limits 21 --change_qkv --relative_position --dist-eval --cfg './experiments/search_space/space-S.yaml' --output_dir './OUTPUT/ETF-space-S-pop8000-seed0-rev'

### 8000 archs xavier
# CUDA_VISIBLE_DEVICES=0, python3 search_autoformer_etf.py --data-path '/dataset/ILSVRC2012' --gp --population-num 8000 \
#  --param-limits 54 --min-param-limits 52 --change_qkv --relative_position --dist-eval --cfg './experiments/search_space/space-B.yaml' --output_dir './OUTPUT/ETF-space-B-pop8000-seed0-rev-xavier'

# CUDA_VISIBLE_DEVICES=1, python3 search_autoformer_etf.py --data-path '/dataset/ILSVRC2012' --gp --population-num 8000 \
#   --param-limits 6 --min-param-limits 4 --change_qkv --relative_position --dist-eval --cfg './experiments/search_space/space-T.yaml' --output_dir './OUTPUT/ETF-space-T-pop8000-seed0-rev-xavier'

# CUDA_VISIBLE_DEVICES=1, python3 search_autoformer_etf.py --data-path '/dataset/ILSVRC2012' --gp --population-num 8000 \
#  --param-limits 23 --min-param-limits 21 --change_qkv --relative_position --dist-eval --cfg './experiments/search_space/space-S.yaml' --output_dir './OUTPUT/ETF-space-S-pop8000-seed0-rev-xavier'

# ### 50000 archs xavier
# CUDA_VISIBLE_DEVICES=0, python3 search_autoformer_etf.py --data-path '/dataset/ILSVRC2012' --gp --population-num 50000 \
#  --param-limits 54 --min-param-limits 52 --change_qkv --relative_position --dist-eval --cfg './experiments/search_space/space-B.yaml' --output_dir './OUTPUT/ETF-space-B-pop50000-seed0-rev-xavier'

# CUDA_VISIBLE_DEVICES=1, python3 search_autoformer_etf.py --data-path '/dataset/ILSVRC2012' --gp --population-num 50000 \
#   --param-limits 6 --min-param-limits 4 --change_qkv --relative_position --dist-eval --cfg './experiments/search_space/space-T.yaml' --output_dir './OUTPUT/ETF-space-T-pop50000-seed0-rev-xavier'

# CUDA_VISIBLE_DEVICES=2, python3 search_autoformer_etf.py --data-path '/dataset/ILSVRC2012' --gp --population-num 50000 \
#  --param-limits 23 --min-param-limits 21 --change_qkv --relative_position --dist-eval --cfg './experiments/search_space/space-S.yaml' --output_dir './OUTPUT/ETF-space-S-pop50000-seed0-rev-xavier'

#### 8000 archs
# CUDA_VISIBLE_DEVICES=0, python3 search_autoformer_etf.py --data-path '/dataset/ILSVRC2012' --gp --population-num 10000 --seed 123 \
#  --param-limits 54 --min-param-limits 52 --change_qkv --relative_position --dist-eval --cfg './experiments/search_space/space-B.yaml' --output_dir './OUTPUT/Search-ETF-cap-trunc-seed123-iter10000/Base'

# CUDA_VISIBLE_DEVICES=1, python3 search_autoformer_etf.py --data-path '/dataset/ILSVRC2012' --gp --population-num 10000 --seed 123 \
#   --param-limits 6 --min-param-limits 4 --change_qkv --relative_position --dist-eval --cfg './experiments/search_space/space-T.yaml' --output_dir './OUTPUT/Search-ETF-cap-trunc-seed123-iter10000/Tiny'

# CUDA_VISIBLE_DEVICES=1, python3 search_autoformer_etf.py --data-path '/dataset/ILSVRC2012' --gp --population-num 10000 --seed 123 \
#  --param-limits 23 --min-param-limits 21 --change_qkv --relative_position --dist-eval --cfg './experiments/search_space/space-S.yaml' --output_dir './OUTPUT/Search-ETF-cap-trunc-seed123-iter10000/Small'

#### 10000 w/o cap archs
# CUDA_VISIBLE_DEVICES=0, python3 search_autoformer_etf.py --data-path '/dataset/ILSVRC2012' --gp --population-num 10000 --seed 0 \
#  --param-limits 54 --min-param-limits 52 --change_qkv --relative_position --dist-eval --cfg './experiments/search_space/space-B.yaml' --output_dir './OUTPUT/Search-ETF-nocap-seed0-iter10000/Base'

CUDA_VISIBLE_DEVICES=1, python3 search_autoformer_etf.py --data-path '/dataset/ILSVRC2012' --gp --population-num 10000 --seed 0 \
  --param-limits 6 --min-param-limits 4 --change_qkv --relative_position --dist-eval --cfg './experiments/search_space/space-T.yaml' --output_dir './OUTPUT/Search-ETF-nocap-seed0-iter10000/Tiny'

CUDA_VISIBLE_DEVICES=1, python3 search_autoformer_etf.py --data-path '/dataset/ILSVRC2012' --gp --population-num 10000 --seed 0 \
 --param-limits 23 --min-param-limits 21 --change_qkv --relative_position --dist-eval --cfg './experiments/search_space/space-S.yaml' --output_dir './OUTPUT/Search-ETF-nocap-seed0-iter10000/Small'