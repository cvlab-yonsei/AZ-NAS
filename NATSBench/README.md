# Credit
- `nats_bench` taken from [NATS-Bench](https://github.com/D-X-Y/NATS-Bench/tree/1d4a304ad1906aa5866563438fcbf0d624b7eda2)
- `xautodl` and `configs` taken from [AutoDL-Projects](https://github.com/D-X-Y/AutoDL-Projects/tree/f46486e21b71ae6459a700be720d7648b5429569)

# Changes in `custom` folder
- `tss_models.py`: Modified from `xaudodl/models/cell_searchs/search_model_random.py`
> *  Add a function that extracts cell features
- `sss_models.py`: Modified from `xaudodl/models/shape_searchs/generic_size_tiny_cell_model.py`; Add a function that extracts cell features
> *  Forward with random one-hot mask prob.
> *  Add a function that extracts cell features