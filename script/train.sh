CUDA_VISIBLE_DEVICES=0 python run.py \
--data_name all_domains \
--n_domains 2 \
--pretrain_domids 0,1 \
--wdomid 0 \
--train_schema cpvae \
--flow_type True \
--styleKL zs \
--sSparsity 0.0001 \
--sJacob_rank 1.0 \
--cSparsity 0.0001 \
--threshold 0 \
--select_k 50 \
--bsz 64 \
--use_pretrainvae 0 \
