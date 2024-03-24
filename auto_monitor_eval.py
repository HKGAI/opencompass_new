import time
import subprocess
import os
import argparse
from pprint import pprint

def get_folder_size(Folderpath):
    size=0
    for path, dirs, files in os.walk(Folderpath):
        for f in files:
            fp = os.path.join(path, f)
            size += os.path.getsize(fp)
    return size

def dump_model_configs(config_list, config_file='hf_llama_7b.py',):
    with open(config_file, 'w') as file:
        print(f'writing {len(config_list)} models to hf_llama_7b.py:')
        pprint(config_list)
        file.write('from opencompass.models import HuggingFaceCausalLM\n\n')
        file.write('models = ' + repr(config_list).replace(", ", ",\n").replace('}','}\n') + '\n')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--serial', action='store_true', help='detect new ckpt after the current evaluation is finished')
    args = parser.parse_args()

    # run with: nohup python -u auto_monitor_eval.py > auto_monitor_eval.master_on_dgx-021.log 2>&1 &
    # run with: nohup python -u auto_monitor_eval.py -s > auto_monitor_eval.master_on_dgx-021.log 2>&1 &

    MODEL_TEMPLATE = {
            'type': 'HuggingFaceCausalLM',
            'abbr': 'exp2.6/20.00B',
            'path': '/workspace/aifs4su/code/checkpoints/hkg_7b_nl_tp2_pp1_mb1_gb1024_gas4/pt2.6/hf_ckpt/20.00B',
            'tokenizer_path': '/workspace/aifs4su/code/checkpoints/hkg_7b_nl_tp2_pp1_mb1_gb1024_gas4/pt2.6/hf_ckpt/hkg_hk50B_hf',
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
        }


    MAXIMUM_RUN_HOUR = 300
    # megatron_ckpt_path = 'checkpoints/hkg_7b_nl_tp2_pp1_mb1_gb1024_gas4/pt2.6/checkpoint'
    hf_ckpt_path = '/workspace/aifs4su/code/checkpoints/hkg_7b_nl_tp2_pp1_mb1_gb1024_gas4/pt2.6/hf_ckpt'
    model_config_file = "/workspace/opencompass_new/hf_llama_7b.py"

    MINIMUM_TOKEN_TO_TEST = 600.0
    MAXIMUM_TOKEN_TO_TEST = float('inf')
    SKIP_TOKEN_TO_TEST = ['959.93B']
    # Record the start time
    start_time = time.time()



    # initial_files = []
    MAX_INIT_TO_ADD = 600.0
    initial_files = os.listdir(hf_ckpt_path)
    initial_files = [
        file for file in initial_files 
        if file.endswith('B') and float(file.replace('B','')) < MAX_INIT_TO_ADD
    ]
    initial_files = sorted(initial_files, key=lambda x: float(x.replace('B','')), reverse=False)

    TESTED_MODELS = []

    print('initial_files:', initial_files)
    while True:
        # watch new generated file in the folder
        current_files = os.listdir(hf_ckpt_path)
        current_files = [
            file for file in current_files 
            if file.endswith('B') and float(file.replace('B',''))>=MINIMUM_TOKEN_TO_TEST and file not in SKIP_TOKEN_TO_TEST
        ]
        # exclude folder that smaller than 10GB
        current_files = [
            file for file in current_files 
            if get_folder_size(os.path.join(hf_ckpt_path, file)) > 12*(1024**3)
        ]
        current_files = sorted(current_files, key=lambda x: float(x.replace('B','')), reverse=False)
        print('current_files:', current_files)
        # Find new files
        new_files = [file for file in current_files if file not in initial_files]
        if len(new_files) > 0:
            print("New checkpoint(s) detected:")
            for file in new_files:
                print(os.path.join(hf_ckpt_path, file))

            model_configs = [] # new detected checkpoints
            for file in new_files:
                ckpt_to_eval = os.path.join(hf_ckpt_path,file)
                # print(f'evaluating {ckpt_to_eval}')
                trained_token = float(file.replace('B',''))
                
                if trained_token < MINIMUM_TOKEN_TO_TEST or trained_token > MAXIMUM_TOKEN_TO_TEST:
                    continue
                else:
                    new_model_conf = MODEL_TEMPLATE.copy()
                    base_abbr = os.path.basename(new_model_conf['abbr'])
                    new_model_conf['abbr'] = new_model_conf['abbr'].replace(base_abbr, file)
                    new_model_conf['path'] = ckpt_to_eval
                    model_configs.append(new_model_conf)
                    # upadte the minimum
                    MINIMUM_TOKEN_TO_TEST = trained_token if trained_token >= MINIMUM_TOKEN_TO_TEST else MINIMUM_TOKEN_TO_TEST

            TESTED_MODELS = TESTED_MODELS + model_configs
            
            print(f'##### Submitted Evaluation on checkpoint(s): #####')
            print("\n".join(new_files))
            # Define your bash command
            # remember to change the config in hkgai/launcher/scripts/pretrain/pt2/batch_convert_ckpt.pt2.6.sh
            # get the current date time
            # bash_command = f"nohup python run.py eval_llama_7b_test.py > {eval_log_file} 2>&1 &"
            # bash_command = f"nohup python test_print.py > {eval_log_file} 2>&1 &"

            # write the new checkpoints to configs
            dump_model_configs(model_configs, 'hf_llama_7b.py')
            if args.serial:
                bash_command = f"python -u run.py eval_llama_7b_test.py -l -r 20240324_101010"
                print(f'run command:', bash_command)
                subprocess.run(bash_command.split())

                print("#"*10 + "\nRe-scan the previous failed evalution\n" + "#"*10)
                dump_model_configs(TESTED_MODELS, 'hf_llama_7b.py')
                bash_command = f"python -u run.py eval_llama_7b_test.py -l -r 20240324_101010"
                print(f'run command:', bash_command)
                subprocess.run(bash_command.split())
            else:
                current_date_time = time.strftime("%Y%m%d-%H%M%S")
                eval_log_file = f'auto_eval_{current_date_time}.log'
                bash_command = f"bash auto_submit.sh {eval_log_file}"
                print(f'run command:', bash_command)
                subprocess.run(bash_command.split())

            # Update the initial file list
            print('update tested file list')
            initial_files = current_files
        

            # Record the end time
            current_time = time.time()
            print(f'The program has run {current_time-start_time} seconds')
            if (current_time - start_time)/3600>=MAXIMUM_RUN_HOUR:
                print(f'exceed maximum run time {MAXIMUM_RUN_HOUR} hour')
                break
        else:
            print('no new file, hang')
            time.sleep(600)

    # mannually run
    # nohup python run.py eval_llama_7b_test.py > eval_659.95B.log 2>&1 &
