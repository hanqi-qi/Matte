# Copyright (c) 2020-present, Royal Bank of Canada.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

"""The test domains are different from pretraining domains"""

import config
import torch 
from utils.text_utils_our import MonoTextData
from models.decomposed_vae import DecomposedVAE
import argparse
import numpy as np
import random
import os
from os import system
from utils.dist_utils import cal_log_density
from models.utils import select_exampls,style_shift

# CUDA_VISIBLE_DEVICES = "1,3"
random.seed(2023)

domain_dict = {"imdb":0,"yelp_dast":1,"amazon":2,"yahoo":3}
domain_i2d = {"0":"imdb","1": "yelp_dast","2":"amazon","3":"yahoo"}
root_path = "../data/"

def load_train_feats():
    feat_all = []
    for domain_name in domain_dict.keys():
        data_pth = root_path+"/%s" %domain_name
        feat_pth = os.path.join(data_pth, "%s_%s.npy" %("train",args.feat))
        feat = np.load(feat_pth)#[N,300]
        feat_all.append(feat)
    output_feats = np.concatenate(([feat for feat in feat_all]))
    return output_feats

def get_coordinates(a, b, p):
    pa = p - a
    ba = b - a
    t = torch.sum(pa * ba) / torch.sum(ba * ba)
    d = torch.norm(pa - t * ba, 2)
    return t, d

def main(args):
    conf = config.CONFIG['yelp']
    pretrain_ids = args.pretrain_domids.split(",")
    print(len(pretrain_ids))
    pretrain_data_name = domain_i2d[pretrain_ids[0]] if len(pretrain_ids)<2 else "all_domains"
    print(pretrain_data_name)
    test_domids = args.test_domids.split(",")
    assert len(test_domids) == 1
    args.data_name = domain_i2d[test_domids[0]]
    print("Train on %s, Evaluate on %s dataset:"%(args.pretrain_domids,args.data_name))

    dev_data_pth = os.path.join(root_path+"%s/"%args.data_name, "test_data.txt")
    dev_feat_pth = os.path.join(root_path+"%s/"%args.data_name, "test_%s.npy" % args.feat)
    dev_feat = np.load(dev_feat_pth)
    test_data_pth = os.path.join(root_path+"%s/"%args.data_name, "test_data.txt")
    test_feat_pth = os.path.join(root_path+"%s/"%args.data_name, "test_%s.npy" % args.feat)
    test_feat = np.load(test_feat_pth)

    #creat vocab from all domains training data.
    if len(pretrain_ids)>1:
        train_data_pth = os.path.join(root_path, "train_all_data.txt")
    else:
        train_data_pth = os.path.join(root_path+"%s/"%pretrain_data_name, "train_data.txt")
    # train_data_pth = os.path.join(root_path, "train_all_data.txt")
    train_data = MonoTextData(train_data_pth,args.n_domains,vocab_size=100000)
    vocab = train_data.vocab
    print('Vocabulary size: %d' % len(vocab))
    train_data_pth = os.path.join(root_path+"%s/"%args.data_name, "train_data.txt")
    train_data = MonoTextData(train_data_pth,args.n_domains,vocab_size=100000)
    #creat training data from specific domain
    train_feat_pth = os.path.join(root_path+"%s/"%args.data_name, "train_%s.npy" % args.feat)
    train_feat = np.load(train_feat_pth)
    train_domid = train_data.domids
    # random_ids = random.choices(range(len(test_data.data)),k=1000)

    dev_data = MonoTextData(dev_data_pth,args.n_domains, vocab=vocab)
    assert len(dev_data) == dev_feat.shape[0]
    
    test_data = MonoTextData(test_data_pth, args.n_domains, vocab=vocab) 
    #Only randomly select 1000 samples for testing
    test_domid = test_data.domids
    test_label = test_data.labels
    test_sent = test_data.data
    assert len(test_sent) == test_feat.shape[0] == len(test_label)==len(test_domid)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    kwargs = {
        "train": ([1], None,None,None),
        "valid": (None, None,None, None),
        "test": (None, None,None,None),
        "feat": None,
        "bsz":64,
        "save_path": None,
        "logging": None,
        "text_only": args.text_only,
        "n_domains":args.wdomid,
        "train_schema":args.train_schema,
        "wdomid":args.wdomid,
        "sSparsity":args.sSparsity,
    }
    
    params = conf["params"]
    #customized different from config
    params["vae_params"]["vocab"] = vocab
    params["vae_params"]["device"] = "cuda"
    params["vae_params"]["text_only"] = args.text_only
    params["vae_params"]["n_domains"] = args.wdomid
    params["vae_params"]["mlp_ni"] = train_feat.shape[1]
    params["vae_params"]["flow_type"] = args.flow_type
    params["vae_params"]["flow_nlayer"] = 1
    params["vae_params"]["flow_dim"] = 16
    params["vae_params"]["sJacob_rank"] = 1
    params["vae_params"]["u_dim"] = args.u_dim
    params["vae_params"]["styleKL"] = args.styleKL
    params["vae_params"]["cSparsity"] = args.cSparsity
    params["vae_params"]["select_k"] = args.select_k
    params["vae_params"]["threshold"] = args.threshold
    params["vae_params"]["start_epoch"] = args.start_epoch
    params["vae_params"]["vae_pretrain"] = 0
    params["vae_params"]["args"]=args

    kwargs = dict(kwargs, **params)
    # with torch.no_grad():
    model = DecomposedVAE(**kwargs)
    model.load(args.load_path,args.test_epoch)
    model.vae.eval()

    # train_data, train_feat, train_domid,_ = train_data.create_data_batch_feats(32, train_feat, device)
    # print("Collecting training distributions...")
    # mus, logvars = [], []
    # step = 0
    # for batch_data, batch_feat,batch_domid in zip(train_data, train_feat,train_domid):
    #     if args.n_domains == 1:
    #         batch_domid = None
    #     # u_embed = model.vae.u_embedding(batch_domid)
    #     mu1, logvar1 = model.vae.lstm_encoder(batch_data,batch_domid)
    #     mu2, logvar2 = model.vae.mlp_encoder(batch_feat,batch_domid)
    #     r, _ = model.vae.mlp_encoder(batch_feat,batch_domid, True)
    #     p = model.vae.get_var_prob(r)
    #     mu = torch.cat([mu1, mu2], -1)
    #     logvar = torch.cat([logvar1, logvar2], -1)
    #     mus.append(mu.detach().cpu())
    #     logvars.append(logvar.detach().cpu())
    #     step += 1
    #     if step % 100 == 0:
    #         torch.cuda.empty_cache()
    # mus = torch.cat(mus, 0)
    # logvars = torch.cat(logvars, 0)

    # #Generate Negative Style Embeddings
    if args.text_only:
        neg_sample = dev_data.data[:10]
        neg_sample_ids = select_exampls(dev_data.data,dev_data.labels,"0",10)
        neg_inputs, _ = dev_data._to_tensor(neg_sample, batch_first=False, device=device)
    else:
        neg_sample_ids,neg_sample_feat,neg_inputids = select_exampls(dev_data.data,dev_feat,dev_data.labels,"0",10,dev_data.vocab,dom_shift=args.dom_shift,test_domid=test_domids[0])
        # neg_sample = dev_feat[:10]#as data are split into negative0 and positive1, so all are negative
        neg_domid = torch.tensor([dev_data.domids[idx] for idx in neg_sample_ids],dtype=torch.long,device=device)
        neg_feat_inputs = torch.tensor(
            neg_sample_feat, dtype=torch.float, requires_grad=False, device=device)
        u_embed = model.vae.u_embedding(neg_domid)
        #get neg_ids
        z1,_ = model.vae.lstm_encoder(torch.tensor(neg_inputids,dtype=torch.long,device=device).transpose(1,0),u=None)
        ori_z2_neg, _,_ = model.vae.encode_semantic(neg_feat_inputs,u_embed,nsamples=1)
        context = model.vae.inject_s2flow(u_embed,ori_z2_neg,z1.unsqueeze(0))
        tilde_z2_neg,_ = model.vae.domain_style(ori_z2_neg,context)
        p = model.vae.get_var_prob(tilde_z2_neg).mean(0,keepdim=True)#[n,n_vars]
        neg_idx = torch.max(p, 1)[1]

    #Generate Positive Style Embeddings
    if args.text_only:
        pos_sample = dev_data.data[-10:]
        pos_inputs, _ = dev_data._to_tensor(pos_sample, batch_first=False, device=device)
    else:
        pos_sample_ids,pos_sample_feat,pos_inputids = select_exampls(dev_data.data,dev_feat,dev_data.labels,"1",10,dev_data.vocab,dom_shift=args.dom_shift,test_domid=test_domids[0])
        pos_domid = torch.tensor([dev_data.domids[idx] for idx in pos_sample_ids],dtype=torch.long,device=device)
        pos_feat_inputs = torch.tensor(
            pos_sample_feat, dtype=torch.float, requires_grad=False, device=device)
        u_embed = model.vae.u_embedding(pos_domid)
        z1,_ = model.vae.lstm_encoder(torch.tensor(pos_inputids,dtype=torch.long,device=device).transpose(1,0),u=None)
        ori_z2_pos, _,_ = model.vae.encode_semantic(pos_feat_inputs,neg_domid)
        context = model.vae.inject_s2flow(u_embed,ori_z2_pos,z1.unsqueeze(0))
        tilde_z2_pos,_ = model.vae.domain_style(ori_z2_pos,context)
        p = model.vae.get_var_prob(tilde_z2_pos).mean(0, keepdim=True)
    top2 = torch.topk(p, 2, 1)[1].squeeze() #select the top2 
    #Double check the positive embedding is exactly Pos
    if top2[0].item() == neg_idx:
        print("Collision!!! Use second most as postive.")
        pos_idx = top2[1].item()
    else:
        pos_idx = top2[0].item()
    
    other_idx = -1
    for i in range(3):
        if i not in [pos_idx, neg_idx]:
            other_idx = i
            break

    print("Negative: %d" % neg_idx)
    print("Positive: %d" % pos_idx)

    bsz = 128
    ori_logps = []
    tra_logps = []
    # pos_z2 = model.vae.mlp_encoder.var_embedding[pos_idx:pos_idx + 1]
    # neg_z2 = model.vae.mlp_encoder.var_embedding[neg_idx:neg_idx + 1]
    # other_z2 = model.vae.mlp_encoder.var_embedding[other_idx:other_idx + 1]
    # _, d0 = get_coordinates(pos_z2[0], neg_z2[0], other_z2[0])
    ori_obs = []
    tra_obs = []
    ref_filename = os.path.join(args.load_path, '%s_reference_%s_results-clip-ep%d-flip%s.txt'%(args.data_name,args.evaluate_type,args.test_epoch,args.inverse_type))
    transfer_filename = os.path.join(os.path.join(args.load_path, '%s_%s_results_%s-clip-ep%d-fliplabel-flip%s.txt'%(args.data_name,args.evaluate_type,args.perturb_type,args.test_epoch,args.inverse_type)))
    ref_f = open(ref_filename, "w")
    # glove_vec, contextual_vec=[],[]
    neutral_num = 0
    with open(transfer_filename, "w") as f:
        idx = 0
        step = 0
        n_samples = len(test_label)
        print(n_samples)
        corrected_flipped = 0
        while idx < n_samples:
            _idx = idx + bsz
            _idx = min(_idx, n_samples)
            labels = test_label[idx:_idx]
            var_id = [pos_idx if label else neg_idx for label in labels]
            text, _ = test_data._to_tensor(
                test_sent[idx:_idx], batch_first=False, device=device)
            feat = torch.tensor(test_feat[idx:_idx], dtype=torch.float, requires_grad=False, device=device)
            domid = torch.tensor(test_domid[idx:_idx], dtype=torch.long, requires_grad=False, device=device)
            u_embed = model.vae.u_embedding(domid)
            z1, _ = model.vae.lstm_encoder(text[:min(text.shape[0], 10)],u=None)
            # z1 = model.vae.causal_influence() * tilde_z1
            ori_z2, _,_ = model.vae.encode_semantic(feat,None,nsamples=1)
            context = model.vae.inject_s2flow(u_embed,ori_z2,z1.unsqueeze(0))#[bs,zc_dim]
            if args.evaluate_type == 'transfer':
                if args.perturb_type == 'flip':

                    tra_z2_noise_list = [torch.mean(tilde_z2_pos,0).unsqueeze(0) if label==0 else torch.mean(tilde_z2_neg,0).unsqueeze(0) for label in labels]
                    tra_z2_noise = torch.concat(tra_z2_noise_list,dim=0)
                    if args.inverse_type == "noise":
                        structured_tilide_z2 =tra_z2_noise
                        tra_z2,_ = model.vae.domain_flow_style.inverse(structured_tilide_z2,context)
                    elif args.inverse_type == "var_embed":
                        structured_tilide_z2 = model.vae.cls_pred(tra_z2_noise)
                        tra_z2,_ = model.vae.domain_flow_style.inverse(structured_tilide_z2,context)
                    elif args.inverse_type == "ori_s":
                        tra_z2= torch.index_select(model.vae.var_embedding, 0, torch.tensor(var_id).cuda())
                    else:
                        print("Invalid perturb type in transfer")
                elif args.perturb_type == 'shift':
                    pos_emb = style_shift(data=dev_data,pos_sample=pos_inputids,pos_feat=pos_feat_inputs,model=model,domid=int(test_domids[0]),device=device)
                    neg_emb = style_shift(data=dev_data,pos_sample=neg_inputids,pos_feat=neg_feat_inputs,model=model,domid=int(test_domids[0]),device=device)
                    shift_emb = torch.concat([neg_emb if label else pos_emb for label in labels],dim=0)
                    # tra_z2 = shift_emb
                    tra_z2 = ori_z2.squeeze(0) + (shift_emb-ori_z2.squeeze(0))*1/3
                else:
                    print("Invalid perturb Type")
            elif args.evaluate_type == "reconstruct":
                if args.inverse_type == "var_embed":
                    tilde_z2,logdet_u2 = model.vae.domain_style(ori_z2,context)
                    structured_tilide_z2 = model.vae.cls_pred(tilde_z2)#(p_noise*var_embed,p_noise)
                    inverse_z2,_ = model.vae.domain_flow_style.inverse(structured_tilide_z2,context)
                    tra_z2 = inverse_z2.squeeze(0)
                elif args.inverse_type=="ori_s":
                    tra_z2 = ori_z2.squeeze(0)
                else:
                    print("Invalid Recon")
            else:
                print("Invalid evaluate type")
                # elif args.inverse_type == "noise":  
            texts = model.vae.decoder.beam_search_decode(z1, tra_z2)
            assert len(texts) == len(test_sent[idx:_idx])
            for label_id,text in enumerate(texts):
                if args.evaluate_type == "transfer":
                    f.write("%d\t%s\n" % (1 - labels[label_id], " ".join(text[1:13])))
                else:
                    f.write("%d\t%s\n" % (labels[label_id], " ".join(text[1:13])))
            # write out the original test samples for comparision
            for label_id,sample in enumerate(test_sent[idx:_idx]):
                ref_f.write("%d\t%s\n" % (labels[label_id], " ".join([test_data.vocab.id2word_[word_id] for word_id in sample])))
                
            ori_z = torch.cat([z1, ori_z2.squeeze(0)], -1)
            # # tra_z = torch.randn_like(ori_z)
            tra_z = torch.cat([z1, tra_z2], -1)
            for i in range(_idx - idx):
                # if args.perturb_type == 'flip' and valid_idx[i].item() is True: #skip the non-flipped samples
                    # continue
                # else:
                # corrected_flipped += 1
                ori_logps.append(cal_log_density(mus, logvars, ori_z[i:i + 1].cpu()))
                tra_logps.append(cal_log_density(mus, logvars, tra_z[i:i + 1].cpu()))

            idx = _idx
            step += 1

            if step % 100 == 0:
                print(step, idx)

    print(ref_filename,"\n",transfer_filename)
    print("Total %d neutral Unchanged sentences"%neutral_num)
def add_args(parser):
    parser.add_argument('--data_name', type=str, default='yelp')
    parser.add_argument('--test_epoch', type=int, default=1)
    parser.add_argument('--feat', type=str, default='glove')
    parser.add_argument('--load_path', type=str,default="")
    parser.add_argument('--text_only', default=False, action='store_true')
    parser.add_argument('--n_domains', default=1, type=int)
    parser.add_argument('--pretrain_domids',type=str,help='domain ids for the pretrain datasets',default="0,1,2,3") #domain_dict = {"imdb":0,"yelp_dast":1,"amazon":2,"yahoo":3}
    parser.add_argument('--test_domids',type=str,help='domain ids for the test datasets',default="3")
    parser.add_argument('--train_schema',type=str,help='cp_vae or sp_vae',default="cp_vae")
    parser.add_argument('--evaluate_type',type=str,help='reconstruct or transfer', default='transfer')
    parser.add_argument('--dom_shift',type=int,help='use sentiment vec from other domains or not', default=0)
    parser.add_argument('--perturb_type',type=str,help='use flip or shift', default="flip")
    parser.add_argument('--inverse_type', default="noise", type=str,help='noise or var_embed')
    parser.add_argument('--u_dim', type=int, default=20,help='the dimension of DomID embedding')
    parser.add_argument('--shift_k',type=int,help='shift n sigma style emb', default=1)
    parser.add_argument('--wdomid', type=int, default=0, help = 'train with label or not')
    parser.add_argument('--sSparsity', type=float, default=0.01, help = 'sparsity weight for s')
    parser.add_argument('--sJacob_rank', type=float, default=1.0, help = 'sparsity constraint on s jacobian, 1 means for all s_dim, otherwise less')
    parser.add_argument('--flow_type',type=str,default="False",help='flow type')
    parser.add_argument('--styleKL', type=str, default="zs", help = 'kl on zs or zs_u')
    #arguments for content variable
    parser.add_argument('--cSparsity', type=float, default=0.0, help ='sparsity weight for c')
    parser.add_argument('--select_k', type=int, default=0, help = 'if penalty the k dimensions in c')
    parser.add_argument('--threshold', type=float, default=0.0, help = 'threshold for selecting c dimension')
    parser.add_argument('--start_epoch',type=int,default=0,help='when start to apply constraints on c')
    parser.add_argument('--lambda_mask', type=float, default=0.001, help ='weight for c mask')
    #arguments for pretrain vae
    parser.add_argument('--do_lower_case',type=int,default=1,help='use lower case or not')
    parser.add_argument('--use_pretrained_model',type=int,default=0,help='whether use pretrain vae')
    parser.add_argument('--latent_size',type=int,default=768,help='latent variable between En/Decoder')
    parser.add_argument('--encoder_model_type',type=str,default="bert-base-uncased",help='encoder for pretrain vae')
    parser.add_argument("--encoder_model_name_or_path", default="bert-base-cased", type=str,
                        help="The encoder model checkpoint for weights initialization.")
    parser.add_argument('--decoder_model_type',type=str,default="gpt2",help='decoder for pretrain vae')
    parser.add_argument("--decoder_model_name_or_path", default="bert-base-cased", type=str,
                        help="The decoder model checkpoint for weights initialization.")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_args(parser)
    args = parser.parse_args()

    main(args)