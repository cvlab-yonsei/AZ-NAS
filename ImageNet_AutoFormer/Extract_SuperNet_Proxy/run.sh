### ETF 8000 samples
# python -m torch.distributed.launch --nproc_per_node=2 --use_env supernet_train.py --data-path /dataset/ILSVRC2012 --gp \
#  --change_qk --relative_position --mode retrain --dist-eval --cfg ./experiments/ETF2/AutoFormer-T.yaml --resume ./supernet-tiny.pth --eval 

# python -m torch.distributed.launch --nproc_per_node=2 --use_env supernet_train.py --data-path /dataset/ILSVRC2012 --gp \
#  --change_qk --relative_position --mode retrain --dist-eval --cfg ./experiments/ETF2/AutoFormer-S.yaml --resume ./supernet-small.pth --eval 

# python -m torch.distributed.launch --nproc_per_node=2 --use_env supernet_train.py --data-path /dataset/ILSVRC2012 --gp \
#  --change_qk --relative_position --mode retrain --dist-eval --cfg ./experiments/ETF2/AutoFormer-B.yaml --resume ./supernet-base.pth --eval 


# # ### TFTAS
# python -m torch.distributed.launch --nproc_per_node=2 --use_env supernet_train.py --data-path /dataset/ILSVRC2012 --gp \
#  --change_qk --relative_position --mode retrain --dist-eval --cfg ./experiments/TFTAS/AutoFormer-T.yaml --resume ./supernet-tiny.pth --eval 

# python -m torch.distributed.launch --nproc_per_node=2 --use_env supernet_train.py --data-path /dataset/ILSVRC2012 --gp \
#  --change_qk --relative_position --mode retrain --dist-eval --cfg ./experiments/TFTAS/AutoFormer-S.yaml --resume ./supernet-small.pth --eval 

# python -m torch.distributed.launch --nproc_per_node=2 --use_env supernet_train.py --data-path /dataset/ILSVRC2012 --gp \
#  --change_qk --relative_position --mode retrain --dist-eval --cfg ./experiments/TFTAS/AutoFormer-B.yaml --resume ./supernet-base.pth --eval 

# ### Autoformer
# python -m torch.distributed.launch --nproc_per_node=2 --use_env supernet_train.py --data-path /dataset/ILSVRC2012 --gp \
#  --change_qk --relative_position --mode retrain --dist-eval --cfg ./experiments/subnet/AutoFormer-T.yaml --resume ./supernet-tiny.pth --eval 

# python -m torch.distributed.launch --nproc_per_node=2 --use_env supernet_train.py --data-path /dataset/ILSVRC2012 --gp \
#  --change_qk --relative_position --mode retrain --dist-eval --cfg ./experiments/subnet/AutoFormer-S.yaml --resume ./supernet-small.pth --eval 

# python -m torch.distributed.launch --nproc_per_node=2 --use_env supernet_train.py --data-path /dataset/ILSVRC2012 --gp \
#  --change_qk --relative_position --mode retrain --dist-eval --cfg ./experiments/subnet/AutoFormer-B.yaml --resume ./supernet-base.pth --eval 

### ETF 10000 samples
python -m torch.distributed.launch --nproc_per_node=2 --use_env supernet_train.py --data-path /dataset/ILSVRC2012 --gp \
 --change_qk --relative_position --mode retrain --dist-eval --cfg ./experiments/ETF-pop10000-seed123/TF_TAS-T.yaml --resume ./supernet-tiny.pth --eval

python -m torch.distributed.launch --nproc_per_node=2 --use_env supernet_train.py --data-path /dataset/ILSVRC2012 --gp \
 --change_qk --relative_position --mode retrain --dist-eval --cfg ./experiments/ETF-pop10000-seed123/TF_TAS-S.yaml --resume ./supernet-small.pth --eval 

python -m torch.distributed.launch --nproc_per_node=2 --use_env supernet_train.py --data-path /dataset/ILSVRC2012 --gp \
 --change_qk --relative_position --mode retrain --dist-eval --cfg ./experiments/ETF-pop10000-seed123/TF_TAS-B.yaml --resume ./supernet-base.pth --eval 