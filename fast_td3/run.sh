CUDA_VISIBLE_DEVICES=1 tsp python -m train \
  --actor egnn \
  --env_name h1-balance_simple-v0 \
  --model_kwargs model_config/egnn.json;

CUDA_VISIBLE_DEVICES=2 tsp python -m train \
  --actor egnn \
  --env_name h1-reach-v0 \
  --model_kwargs model_config/egnn.json;
