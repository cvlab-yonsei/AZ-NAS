<!-- # Install
```bash
pip install timm thop einops
``` -->

# Credit
- The search space is proposed in [AutoFormer](https://github.com/microsoft/Cream/tree/b799630a29995163f282b15e2f38701160272fd1/AutoFormer)
- Code taken from [TFTAS](https://github.com/decemberzhou/TF_TAS/tree/42616bcf1b6bb643bf968a8342f8aaddc4f53f32)

# Change notes
- `./model/autoformer_space.py`
> * Add a function that extracts features from primary blocks
> * Turn off Automatic Mixed Precision (AMP) when the NaN loss error is raised

- `./lib/training_free/indicators/az_nas.py`
> * Add our algorithm

- `search_autoformer_az.py`: Modified from [search_autoformer.py](https://github.com/decemberzhou/TF_TAS/blob/42616bcf1b6bb643bf968a8342f8aaddc4f53f32/search_autoformer.py)
> * Search with zero-cost proxies of AZ-NAS
> * Add non-linear ranking aggregation

- `train.py`
> * Add tensorboard logging