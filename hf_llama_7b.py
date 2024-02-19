from opencompass.models import HuggingFaceCausalLM

# 支持多个模型一起测评
models = [
    {
        'type': 'HuggingFaceCausalLM',
        'abbr': 'exp2.1/1283.46B',
        'path': '/workspace/code/opencompass/checkpoints/hkg_7b_nl_tp1_pp1_mb1_gb512_gas2/exp2.1/hf_ckpt/1283.46B',
        'tokenizer_path': '/workspace/code/Megatron-LM/hkgai/hf/hkg_amber_hf',
        'tokenizer_kwargs': {
            'padding_side': 'left',
            'truncation_side': 'left',
            'use_fast': False,
            'trust_remote_code': True,
        },
        'max_out_len': 100,
        'max_seq_len': 4096,
        'batch_size': 16,
        'model_kwargs': {
            'device_map': 'auto', 
            'trust_remote_code': True,
        },
        'batch_padding': False,
        'run_cfg': {'num_gpus': 1, 'num_procs': 1},
    },
    # {
    #     'type': 'HuggingFaceCausalLM',
    #     'abbr': 'slimpajama/hf_096000',
    #     'path': '/workspace/code/hf_ckpt/slimpajama/hf_096000',
    #     'tokenizer_path': 'baichuan-inc/Baichuan2-7B-Base',
    #     'tokenizer_kwargs': {
    #         'padding_side': 'left',
    #         'truncation_side': 'left',
    #         'use_fast': False,
    #         'trust_remote_code': True,
    #     },
    #     'max_out_len': 100,
    #     'max_seq_len': 2048,
    #     'batch_size': 16,
    #     'model_kwargs': {
    #         'device_map': 'auto', 
    #         'trust_remote_code': True,
    #     },
    #     'batch_padding': False,
    #     'run_cfg': {'num_gpus': 1, 'num_procs': 1},
    # },
]

