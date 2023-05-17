#!/bin/bash
# CUDA_VISIBLE_DEVICES=0, python3 search_autoformer_etf.py --data-path '/dataset/ILSVRC2012' --gp \
#  --param-limits 54 --change_qk --relative_position --dist-eval --cfg './experiments/search_space/space-B.yaml' --output_dir './OUTPUT/ETF-space-B'

# CUDA_VISIBLE_DEVICES=1, python3 search_autoformer_etf.py --data-path '/dataset/ILSVRC2012' --gp \
#  --param-limits 23 --change_qk --relative_position --dist-eval --cfg './experiments/search_space/space-S.yaml' --output_dir './OUTPUT/ETF-space-S'

# CUDA_VISIBLE_DEVICES=1, python3 search_autoformer_etf.py --data-path '/dataset/ILSVRC2012' --gp \
#   --param-limits 6 --change_qk --relative_position --dist-eval --cfg './experiments/search_space/space-T.yaml' --output_dir './OUTPUT/ETF-space-T'


### 50000 archs

# CUDA_VISIBLE_DEVICES=0, python3 search_autoformer_etf.py --data-path '/dataset/ILSVRC2012' --gp --population-num 50000 \
#  --param-limits 54 --change_qk --relative_position --dist-eval --cfg './experiments/search_space/space-B.yaml' --output_dir './OUTPUT/ETF-space-B-pop50000'

CUDA_VISIBLE_DEVICES=1, python3 search_autoformer_etf.py --data-path '/dataset/ILSVRC2012' --gp --population-num 50000 \
 --param-limits 23 --change_qk --relative_position --dist-eval --cfg './experiments/search_space/space-S.yaml' --output_dir './OUTPUT/ETF-space-S-pop50000'

CUDA_VISIBLE_DEVICES=1, python3 search_autoformer_etf.py --data-path '/dataset/ILSVRC2012' --gp --population-num 50000 \
  --param-limits 6 --change_qk --relative_position --dist-eval --cfg './experiments/search_space/space-T.yaml' --output_dir './OUTPUT/ETF-space-T-pop50000'