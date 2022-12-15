# CUDA_VISIBLE_DEVICES=0,1 \
# python RSPS_cell_level.py  --config_path "./test_search.config" \
#                            --train_config_path "./test_train.config" \
#                            --save_dir "./results/test"

CUDA_VISIBLE_DEVICES=0,1 \
python RSPS_cell_level.py  --config_path "./test_search.config" \
                           --train_config_path "./test_train.config" \
                           --diverse_metrics True \
                           --save_dir "./results/test_diverse_metrics"