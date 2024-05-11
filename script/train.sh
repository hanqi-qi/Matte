CUDA_VISIBLE_DEVICES=1 python run.py \
--data_name yelp \
--n_domains 4 \
--pretrain_domids 0,1,2,3 \
--wdomid 0 \
--train_schema cpvae \
--flow_type True \
--seed 39 \