test_domids=$1
CUDA_VISIBLE_DEVICES=1 python transfer_spdom.py \
--load_path /mnt/Data3/hanqiyan/style_transfer_baseline/checkpoint/model-all_domains-glove-w0domid-jointtrain/20230331-204406 \
--pretrain_domids 0,1,2,3 \
--test_domids $test_domids \
--evaluate_type transfer \
--dom_shift 0 \
--perturb_type flip \
--wdomid 0 \
--shift_k 1 \