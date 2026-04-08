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


python -m train --actor transformer --env_name h1-slide-v1 --model_kwargs model_config/transformer.json;
python -m train --actor transformer --env_name h1-balance_simple-v1 --model_kwargs model_config/transformer.json;
python -m train --actor transformer --env_name h1-slide-v1 --model_kwargs model_config/transformer2.json;
python -m train --actor transformer --env_name h1-balance_simple-v1 --model_kwargs model_config/transformer2.json;
python -m train --actor transformer --env_name h1-slide-v1 --model_kwargs model_config/transformer3.json;
python -m train --actor transformer --env_name h1-balance_simple-v1 --model_kwargs model_config/transformer3.json;
python -m train --actor transformer --env_name h1-slide-v1 --model_kwargs model_config/transformer4.json;
python -m train --actor transformer --env_name h1-balance_simple-v1 --model_kwargs model_config/transformer4.json;
python -m train --actor transformer --env_name h1-slide-v1 --model_kwargs model_config/transformer5.json;
python -m train --actor transformer --env_name h1-balance_simple-v1 --model_kwargs model_config/transformer5.json;


python -m train --actor transformer_v2 --env_name h1-push-v1 --model_kwargs model_config/transformer_v2.json;
python -m train --actor transformer_v2 --env_name h1-reach-v1 --model_kwargs model_config/transformer_v2.json;

python -m train --actor mlp --env_name h1hand-run-v1;
python -m train --actor mlp --env_name h1hand-slide-v1;
python -m train --actor mlp --env_name h1hand-balance_simple-v1;

CUDA_VISIBLE_DEVICES=2 python -m train --actor mlp --env_name h1hand-push-v1;
CUDA_VISIBLE_DEVICES=2 python -m train --actor mlp --env_name h1hand-reach-v1;
CUDA_VISIBLE_DEVICES=2 python -m train --actor mlp --env_name h1hand-basketball-v1;
CUDA_VISIBLE_DEVICES=2 python -m train --actor mlp --env_name h1hand-door-v1;
CUDA_VISIBLE_DEVICES=2 python -m train --actor mlp --env_name h1hand-window-v1;
CUDA_VISIBLE_DEVICES=2 python -m train --actor mlp --env_name h1hand-balance_hard-v1;

