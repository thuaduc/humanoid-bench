python -m train --actor egnn_v3 --env_name h1-run-v1 --model_kwargs model_config/egnn.json;
python -m train --actor egnn_v5 --env_name h1-balance_simple-v1 --model_kwargs model_config/egnn.json;
python -m train --actor egnn_v3 --env_name h1-push-v1 --model_kwargs model_config/egnn.json;