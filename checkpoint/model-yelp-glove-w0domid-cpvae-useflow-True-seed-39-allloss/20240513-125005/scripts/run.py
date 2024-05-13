from utils.exp_utils import create_exp_dir
from utils.text_utils_our import MonoTextData, construct_domain_datasets,BucketingDataLoader
import argparse
import os
import torch
import time
import config
from models.decomposed_vae import DecomposedVAE
from models.configure_prevae import configure_prevae
import numpy as np
import warnings

warnings.filterwarnings("ignore")
import logging
logging.getLogger('transformers.tokenization_utils').setLevel(logging.ERROR)

domain_dict = {"imdb":0,"yelp_dast":1,"amazon":2,"yahoo":3}
domain_i2d = {"0":"imdb","1": "yelp_dast","2":"amazon","3":"yahoo"}

def main(args):
    conf = config.CONFIG['yelp']
    pretrain_domids = args.pretrain_domids.split(",")
    args.n_domains = len(pretrain_domids)

    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_data_pth, dev_data_pth, test_data_pth, train_feat, dev_feat, test_feat = construct_domain_datasets(pretrain_domids=pretrain_domids, feat_type=args.feat)
    
    if not args.use_pretrainvae: 
        train_data = MonoTextData(train_data_pth,args.n_domains)    
        assert len(train_data) == train_feat.shape[0]
        vocab = train_data.vocab
        print('Vocabulary size: %d' % len(vocab))
        dev_data = MonoTextData(test_data_pth,args.n_domains, vocab=vocab)
        assert len(dev_data) == dev_feat.shape[0]
        test_data = MonoTextData(test_data_pth, args.n_domains, vocab=vocab)  
        assert len(test_data) == test_feat.shape[0]
        
        train = train_data.create_data_batch_feats(args.bsz, train_feat, args.device)
        dev = dev_data.create_data_batch_feats(args.bsz, dev_feat, args.device)
        test = test_data.create_data_batch_feats(args.bsz, test_feat, args.device)
        feat = train_feat
    else:
        encoder, decoder,encoder_tokenizer, decoder_tokenizer = configure_prevae(
            args.encoder_model_type, args.encoder_model_name_or_path,
            args.decoder_model_type, args.decoder_model_name_or_path,
            args.latent_size,args.save,args.load_weight,args.device)
        
        train = BucketingDataLoader(train_data_pth, args.bsz, args.max_seq_length, encoder_tokenizer, decoder_tokenizer, args, bucket=100, shuffle=False) 
        dev = BucketingDataLoader(dev_data_pth, args.bsz, args.max_seq_length, encoder_tokenizer, decoder_tokenizer, args, bucket=100, shuffle=False) 
        test = BucketingDataLoader(test_data_pth, args.bsz, args.max_seq_length, encoder_tokenizer, decoder_tokenizer, args, bucket=100, shuffle=False)
        feat = None
        vocab = encoder_tokenizer.vocab

    save_path = '{}-{}-{}-useflow-{}-sSpaweight-{}-cSpaweight:{}-usePretrainVAE:{}-latentsize:{}'\
        .format(args.save, args.data_name,args.train_schema,args.flow_type,args.sSparsity,args.cSparsity,args.use_pretrainvae,args.latent_size)
    args.save = save_path
    print(f"save to {save_path}")
    save_path = os.path.join(save_path, time.strftime("%Y%m%d-%H%M%S"))
    scripts_to_save = [
        'run.py', 'models/decomposed_vae_flow.py', 'models/influential_vae_fl.py',
        'models/base_network.py', 'config.py']
    logging = create_exp_dir(save_path, scripts_to_save=scripts_to_save,
                             debug=args.debug)
    kwargs = {
        "train": train,
        "valid": dev,
        "test": test,
        "feat": feat,
        "bsz": args.bsz,
        "save_path": save_path,
        "logging": logging,
        "text_only": args.text_only,
        "n_domains":args.wdomid,
        "train_schema":args.train_schema,
        "wdomid":args.wdomid,
        "sSparsity":args.sSparsity,
        "cSparsity":args.cSparsity,
        "lambda_mask":args.lambda_mask,
    }
    
    params = conf["params"]
    #customized different from config
    params["vae_params"]["vocab"] = vocab
    params["vae_params"]["device"] = args.device
    params["vae_params"]["text_only"] = args.text_only
    params["vae_params"]["n_domains"] = args.wdomid
    params["vae_params"]["mlp_ni"] = train_feat.shape[1]
    params["vae_params"]["flow_type"] = args.flow_type
    params["vae_params"]["flow_nlayer"] = 1
    params["vae_params"]["flow_dim"] = 16
    params["vae_params"]["sJacob_rank"] = args.sJacob_rank
    params["vae_params"]["u_dim"] = args.u_dim
    params["vae_params"]["styleKL"] = args.styleKL
    params["vae_params"]["cSparsity"] = args.cSparsity
    params["vae_params"]["select_k"] = args.select_k
    params["vae_params"]["threshold"] = args.threshold
    params["vae_params"]["start_epoch"] = args.start_epoch
    params["vae_params"]["vae_pretrain"] = 0
    params["vae_params"]["args"]=args
    
    kwargs = dict(kwargs,**params)

    # encoder,decoder,encoder_tokenizer,decoder_tokenizer = None, None, None, None
    model = DecomposedVAE(**kwargs)
    if args.load_weight:
        pretrained_path = "/home/u2048587/style_transfer_baseline/checkpoint"
        model.load(pretrained_path,0)
        logging("Load Pretrained Model!")
    try:
        valid_loss = model.fit()
        logging("val loss : {}".format(valid_loss))
    except KeyboardInterrupt:
        logging("Exiting from training early")

    model.load(save_path)
    logging("Load Pretrained Model!")
    if args.use_pretrainvae:
        test_loss = model.evaluate_our(model.test_dataloader)
    else:
        test_loss = model.evaluate_our(model.test_data, model.test_feat,model.test_domid)
    logging("test loss: {}".format(test_loss[0]))
    logging("test recon: {}".format(test_loss[1]))
    logging("test kl1: {}".format(test_loss[2]))
    logging("test kl2: {}".format(test_loss[3]))
    logging("test mi1: {}".format(test_loss[4]))
    logging("test mi2: {}".format(test_loss[5]))

def add_args(parser):
    parser.add_argument('--data_name', type=str, default='yelp',help='data name')
    parser.add_argument('--save', type=str, default='checkpoint/model',help='directory name to save')
    parser.add_argument('--bsz', type=int, default=2, help='batch size for training')
    parser.add_argument('--eval_batch_size', type=int, default=16, help='batch size for training')
    parser.add_argument('--text_only', default=False, action='store_true',help='use text only without feats')
    parser.add_argument('--debug', default=False, action='store_true',help='enable debug mode')
    parser.add_argument('--n_domains', default=4, type=int, help='if use multi-domain dataset')
    parser.add_argument('--max_seq_length', default=20, type=int, help='the maximum length of sentence')
    parser.add_argument('--pretrain_domids',type=str,help='domain ids for the pretrain datasets',default="0,1,2,3") #domain_dict = {"imdb":0,"yelp_dast":1,"amazon":2,"yahoo":3}
    parser.add_argument('--test_domids',type=str,help='domain ids for the test datasets',default="0,1,2,3")
    parser.add_argument('--device', type=str, default='cpu',help='use cpu or gpu')
    parser.add_argument('--feat', type=str, default='glove',help='feat repr')
    parser.add_argument('--u_dim', type=int, default=20,help='the dimension of DomID embedding')
    parser.add_argument('--train_schema',type=str,help='inDomain or joint or cpvae',default="cpvae")
    parser.add_argument('--load_weight',action='store_true',default=False,help='load pretrained weight')
    parser.add_argument('--flow_type',type=str,default="True",help='flow type')
    #arguments for style variable
    parser.add_argument('--wdomid', type=int, default=0, help = 'train with domid in input or not')
    parser.add_argument('--sSparsity', type=float, default=0.01, help = 'sparsity weight for s')
    parser.add_argument('--sJacob_rank', type=float, default=1.0, help = 'sparsity constraint on s jacobian, 1 means for all s_dim, otherwise less')
    parser.add_argument('--styleKL', type=str, default="zs", help = 'kl on zs or zs_u')
    #arguments for content variable
    parser.add_argument('--cSparsity', type=float, default=0.0, help ='sparsity weight for partial')
    parser.add_argument('--lambda_mask', type=float, default=0.0, help ='sparsity weight for c-mask')
    parser.add_argument('--select_k', type=int, default=0, help = 'if penalty the k dimensions in c')
    parser.add_argument('--threshold', type=float, default=0.0, help = 'threshold for selecting c dimension')
    parser.add_argument('--start_epoch',type=int,default=0,help='when start to apply constraints on c')
    #arguments for pretrain vae
    parser.add_argument('--do_lower_case',type=int,default=1,help='use lower case or not')
    parser.add_argument('--use_pretrainvae',type=int,default=1,help='whether use pretrain vae')
    parser.add_argument('--latent_size',type=int,default=128,help='latent variable between En/Decoder')
    parser.add_argument('--encoder_model_type',type=str,default="bert",help='encoder for pretrain vae')
    parser.add_argument("--encoder_model_name_or_path", default="bert-base-uncased", type=str,
                        help="The encoder model checkpoint for weights initialization.")
    parser.add_argument('--decoder_model_type',type=str,default="gpt2",help='decoder for pretrain vae')
    parser.add_argument("--decoder_model_name_or_path", default="gpt2", type=str,
                        help="The decoder model checkpoint for weights initialization.")
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_args(parser)
    args = parser.parse_args()
    main(args)