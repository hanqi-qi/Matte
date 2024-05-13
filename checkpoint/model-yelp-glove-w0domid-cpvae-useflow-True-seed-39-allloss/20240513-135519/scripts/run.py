# Copyright (c) 2020-present, Royal Bank of Canada.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from utils.exp_utils import create_exp_dir
from utils.text_utils_our import MonoTextData
import argparse
import os
import torch
import time
import config
from models.decomposed_vae import DecomposedVAE
import numpy as np
import random
from os import system

domain_dict = {"imdb":0,"yelp_dast":1,"amazon":2,"yahoo":3}
domain_i2d = {"0":"imdb","1": "yelp_dast","2":"amazon","3":"yahoo"}
root_path = "/mnt/Data3/hanqiyan/UDA/real_world/data/"

def main(args):
    #set random seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    # conf = config.CONFIG["all_pretrain"]
    conf = config.CONFIG['yelp']
    pretrain_domids = args.pretrain_domids.split(",")
    args.n_domains = len(pretrain_domids)

    #load multiple domain datasets or single-domain
    do_dataset = True
    if args.n_domains > 1:
        print("Training on %s domains"%args.pretrain_domids)
        data_files_command = "cat"
        feat_all_splits = []
        for split in ["train","dev","test"]:
            if os.path.exists("/mnt/Data3/hanqiyan/UDA/real_world/data/%s_all_data.txt"%split):
                print("%s dataset is built"%split)
                do_dataset = False
            data_files_command = "cat"
            feat_all = []
            for domid in pretrain_domids:
                data_name = domain_i2d[domid]
                data_pth = "/mnt/Data3/hanqiyan/UDA/real_world/data/%s" %data_name
                feat_pth = os.path.join(data_pth, "%s_%s.npy" %(split,args.feat))
                feat = np.load(feat_pth)#[N,300]
                feat_all.append(feat)
            feat_all_splits.append(np.concatenate(([feat for feat in feat_all])))
            # concate the different pretrained datasets
            if do_dataset:
                data_files_command += " > /mnt/Data3/hanqiyan/UDA/real_world/data/%s_all_data.txt"%split
                system(data_files_command)
            train_data_pth = os.path.join(root_path, "train_all_data.txt")
            dev_data_pth = os.path.join(root_path, "dev_all_data.txt")
            test_data_pth = os.path.join(root_path, "test_all_data.txt")
        train_feat = feat_all_splits[0]
        dev_feat = feat_all_splits[1]
        test_feat = feat_all_splits[2]
    else:
        args.data_name = domain_i2d[pretrain_domids[0]]
        train_data_pth = os.path.join(root_path+"%s/"%args.data_name, "train_data.txt")
        train_feat_pth = os.path.join(root_path+"%s/"%args.data_name, "train_%s.npy" % args.feat)
        train_feat = np.load(train_feat_pth)
        dev_data_pth = os.path.join(root_path+"%s/"%args.data_name, "dev_data.txt")
        dev_feat_pth = os.path.join(root_path+"%s/"%args.data_name, "dev_%s.npy" % args.feat)
        dev_feat = np.load(dev_feat_pth)
        test_data_pth = os.path.join(root_path+"%s/"%args.data_name, "test_data.txt")
        test_feat_pth = os.path.join(root_path+"%s/"%args.data_name, "test_%s.npy" % args.feat)
        test_feat = np.load(test_feat_pth)

    train_data = MonoTextData(train_data_pth,args.n_domains)
    assert len(train_data) == train_feat.shape[0]

    vocab = train_data.vocab
    print('Vocabulary size: %d' % len(vocab))

    dev_data = MonoTextData(dev_data_pth,args.n_domains, vocab=vocab)
    assert len(dev_data) == dev_feat.shape[0]
    
    test_data = MonoTextData(test_data_pth, args.n_domains, vocab=vocab)  
    assert len(test_data) == test_feat.shape[0]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    save_path = '{}-{}-{}-w{}domid-{}-useflow-{}-seed-{}-allloss'.format(args.save, args.data_name, args.feat, args.wdomid,args.train_schema,args.flow_type,args.seed)
    save_path = os.path.join(save_path, time.strftime("%Y%m%d-%H%M%S"))
    scripts_to_save = [
        'run.py', 'models/decomposed_vae.py', 'models/vae.py',
        'models/base_network.py', 'config.py']
    logging = create_exp_dir(save_path, scripts_to_save=scripts_to_save,
                             debug=args.debug)

    if args.text_only:
        train = train_data.create_data_batch(args.bsz, device)
        dev = dev_data.create_data_batch(args.bsz, device)
        test = test_data.create_data_batch(args.bsz, device)
        feat = train
    else:
        train = train_data.create_data_batch_feats(args.bsz, train_feat, device)
        dev = dev_data.create_data_batch_feats(args.bsz, dev_feat, device)
        test = test_data.create_data_batch_feats(args.bsz, test_feat, device)
        feat = train_feat
    train_pos_idx = np.random.choice(feat.shape[0],100)
    train_neg_idx = np.random.choice(10000,100)

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
    params["vae_params"]["vocab"] = vocab
    params["vae_params"]["device"] = device
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
    # params = conf["params"]
    # params["text_only"] = args.text_only
    # params["n_domains"] = args.wdomid
    # params["train_schema"] = args.train_schema
    # params["wdomid"] = args.wdomid
    # params["vae_params"]["vocab"] = vocab
    # params["vae_params"]["device"] = device
    # params["vae_params"]["text_only"] = args.text_only
    # params["vae_params"]["n_domains"] = args.wdomid
    # params["vae_params"]["mlp_ni"] = train_feat.shape[1]
    # params["vae_params"]["flow_type"] = args.flow_type
    # params["vae_params"]["flow_nlayer"] = 1
    # params["vae_params"]["flow_dim"] = 16

    kwargs = dict(kwargs, **params)

    model = DecomposedVAE(**kwargs)
    if args.load_weight:
        pretrained_path = "/mnt/Data3/hanqiyan/style_transfer/checkpoint/cpvae_pretrain-DoCoGen_review-glove/20230128-153150"
        model.load(pretrained_path)
        logging("Load Pretrained Model!")
    try:
        valid_loss = model.fit()
        logging("val loss : {}".format(valid_loss))
    except KeyboardInterrupt:
        logging("Exiting from training early")

    model.load(save_path)
    logging("Load Pretrained Model!")
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
    parser.add_argument('--bsz', type=int, default=32, help='batch size for training')
    parser.add_argument('--text_only', default=False, action='store_true',help='use text only without feats')
    parser.add_argument('--debug', default=False, action='store_true',help='enable debug mode')
    parser.add_argument('--n_domains', default=6, type=int, help='if use multi-domain dataset')
    parser.add_argument('--pretrain_domids',type=str,help='domain ids for the pretrain datasets',default="0,1,2,3") #domain_dict = {"imdb":0,"yelp_dast":1,"amazon":2,"yahoo":3}
    parser.add_argument('--sSparsity', type=float, default=0.01, help = 'sparsity weight for s')
    parser.add_argument('--sJacob_rank', type=float, default=1.0, help = 'sparsity constraint on s jacobian, 1 means for all s_dim, otherwise less')
    parser.add_argument('--styleKL', type=str, default="zs", help = 'kl on zs or zs_u')
    #arguments for content variable
    parser.add_argument('--u_dim', type=int, default=20,help='the dimension of DomID embedding')
    parser.add_argument('--cSparsity', type=float, default=0.0, help ='sparsity weight for partial')
    parser.add_argument('--lambda_mask', type=float, default=0.0, help ='sparsity weight for c-mask')
    parser.add_argument('--select_k', type=int, default=0, help = 'if penalty the k dimensions in c')
    parser.add_argument('--threshold', type=float, default=0.0, help = 'threshold for selecting c dimension')
    parser.add_argument('--start_epoch',type=int,default=0,help='when start to apply constraints on c')
    #arguments for pretrain vae
    parser.add_argument('--test_domids',type=str,help='domain ids for the test datasets',default="0,1,2,3")
    parser.add_argument('--feat', type=str, default='glove',help='feat repr')
    parser.add_argument('--train_schema',type=str,help='inDomain or joint or cpvae',default="inDomain")
    parser.add_argument('--load_weight',action='store_true',default=False,help='load pretrained weight')
    parser.add_argument('--flow_type',type=str,default="ddsf",help='flow type')
    parser.add_argument('--wdomid', type=int, default=0, help = 'train with domid in input or not')
    parser.add_argument('--seed', type=int, default=2023)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_args(parser)
    args = parser.parse_args()
    main(args)