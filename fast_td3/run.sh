python -m train --actor transformer --env_name h1-run-v1 --model_kwargs model_config/transformer.json;
python -m train --actor transformer --env_name h1-slide-v1 --model_kwargs model_config/transformer.json;
python -m train --actor transformer --env_name h1-balance_simple-v1 --model_kwargs model_config/transformer.json;
python -m train --actor transformer --env_name h1-push-v1 --model_kwargs model_config/transformer.json;
python -m train --actor transformer --env_name h1-reach-v1 --model_kwargs model_config/transformer.json;

python -m train --actor mlp --env_name h1-run-v1;
python -m train --actor mlp --env_name h1-slide-v1;
python -m train --actor mlp --env_name h1-balance_simple-v1;
python -m train --actor mlp --env_name h1-push-v1;
python -m train --actor mlp --env_name h1-reach-v1;