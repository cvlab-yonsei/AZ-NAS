# Credit
- Code taken from [TFTAS](https://github.com/decemberzhou/TF_TAS/tree/42616bcf1b6bb643bf968a8342f8aaddc4f53f32)

# Change notes
- `./model/autoformer_space.py`
> * Add a function that extracts block features

- `./lib/training_free/indicators/etf.py`
> * Add our algorithm

- `search_autoformer_etf.py`: Modified from `search_autoformer.py`
> * Search with EFT metrics
> * Add rank aggregation of multiple metrics

- `.sh` files
> * Add parameter constraints

- `train.py`
> * Add tensorboard logging