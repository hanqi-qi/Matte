test_domids=$1
load_path=$2
test_epoch=$3
CUDA_VISIBLE_DEVICES=0 python transfer_spdom.py \
--load_path $load_path \
--pretrain_domids 1 \
--test_domids $test_domids \
--evaluate_type transfer \
--dom_shift 0 \
--flow_type True \
--perturb_type shift \
--wdomid 0 \
--shift_k 1 \
--inverse_type ori_s \
--test_epoch $test_epoch \