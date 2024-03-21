eval_log_file=$1
# nohup python test_print.py > ${eval_log_file} 2>&1 &
nohup python run.py eval_llama_7b_test.py > ${eval_log_file} 2>&1 &