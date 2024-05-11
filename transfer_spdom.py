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
random.seed(8889)

domain_dict = {"imdb":0,"yelp":1,"amazon":2,"yahoo":3,"yelp":4}
domain_i2d = {"0":"imdb","1": "yelp","2":"amazon","3":"yahoo","4":"yelp"}
root_path = "/mnt/Data3/hanqiyan/UDA/real_world/data/"
tense_path = "/mnt/Data3/hanqiyan/style_transfer_baseline/tense_transfer/"


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
    pretrain_data_name = domain_i2d[pretrain_ids[0]] if len(pretrain_ids)<2 else "all_domains"
    print(pretrain_data_name)
    test_domids = args.test_domids.split(",")
    assert len(test_domids) == 1
    args.data_name = domain_i2d[test_domids[0]]
    print("Train on %s, Evaluate on %s dataset:"%(args.pretrain_domids,args.data_name))

    # dev_data_pth = os.path.join(tense_path+"%s/"%args.data_name, "test_reference.txt")
    # dev_feat_pth = os.path.join(tense_path+"%s/"%args.data_name, "test_%s.npy" % args.feat)
    # dev_feat = np.load(dev_feat_pth)
    test_data_pth = os.path.join(tense_path+"%s/"%args.data_name, "test_reference.txt")
    test_feat_pth = os.path.join(root_path+"%s/"%args.data_name, "test_%s.npy" % args.feat)
    test_feat = np.load(test_feat_pth)
    dev_feat = test_feat

    #creat vocab from all domains training data.
    if len(pretrain_ids)>1:
        train_data_pth = os.path.join(root_path, "train_all_data.txt")
        print("use all-domain data to create vocabulary")
    else:
        train_data_pth = os.path.join(root_path+"%s/"%pretrain_data_name, "train_data.txt")
    # train_data_pth = os.path.join(root_path, "train_all_data.txt")
    
    train_data = MonoTextData(train_data_pth,args.n_domains)
    vocab = train_data.vocab
    print('Vocabulary size: %d' % len(vocab))

    dev_data = MonoTextData(test_data_pth,args.n_domains, vocab=vocab)
    assert len(dev_data) == test_feat.shape[0]
    
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
        "pos_list": None,
        "neg_list": None,
        "bsz": 32,
        "save_path": args.load_path,
        "logging": None,
        "text_only": args.text_only,
        "n_domains":args.n_domains-1,
        "train_schema":args.train_schema,
        "wdomid":args.wdomid,
    }
    params = conf["params"]
    params["vae_params"]["vocab"] = vocab
    params["vae_params"]["device"] = device
    params["vae_params"]["text_only"] = args.text_only
    params["vae_params"]["mlp_ni"] = dev_feat.shape[1]
    params["vae_params"]["n_domains"] = 0
    params["vae_params"]["flow_type"] = "ddsf"
    params["vae_params"]["flow_nlayer"] = 1
    params["vae_params"]["flow_dim"] = 8

    kwargs = dict(kwargs, **params)
    model = DecomposedVAE(**kwargs)
    model.load(args.load_path)
    model.vae.eval()

    # #Generate Negative Style Embeddings

    neg_sample_ids,neg_sample_feat,neg_inputids = select_exampls(dev_data.data,dev_feat,dev_data.labels,"0",10,dev_data.vocab,dom_shift=args.dom_shift,test_domid=test_domids[0])
    # neg_sample = dev_feat[:10]#as data are split into negative0 and positive1, so all are negative
    neg_domid = [dev_data.domids[idx] for idx in neg_sample_ids]
    neg_feat_inputs = torch.tensor(
        neg_sample_feat, dtype=torch.float, requires_grad=False, device=device)
    # np.save("./tense_transfer/yelp/neg_noise_emb",neg_feat_inputs.detach().cpu().numpy())
    r, _ = model.vae.mlp_encoder(neg_feat_inputs,neg_domid, True)#[N,dim]
    p = model.vae.get_var_prob(r).mean(0, keepdim=True)#[1,n_vars]
    neg_idx = torch.max(p, 1)[1].item()#select from the n_vars

    pos_sample_ids,pos_sample_feat,pos_inputids = select_exampls(dev_data.data,dev_feat,dev_data.labels,"1",10,dev_data.vocab,dom_shift=args.dom_shift,test_domid=test_domids[0])
    pos_domid = [dev_data.domids[idx] for idx in pos_sample_ids]
    pos_feat_inputs = torch.tensor(
        pos_sample_feat, dtype=torch.float, requires_grad=False, device=device)
    # np.save("./tense_transfer/yelp/pos_noise_emb",pos_feat_inputs.detach().cpu().numpy())
    r, _ = model.vae.mlp_encoder(pos_feat_inputs,pos_domid, True)
    p = model.vae.get_var_prob(r).mean(0, keepdim=True)

    top2 = torch.topk(p, 2, 1)[1].squeeze()
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
    pos_z2 = model.vae.mlp_encoder.var_embedding[pos_idx:pos_idx + 1]
    neg_z2 = model.vae.mlp_encoder.var_embedding[neg_idx:neg_idx + 1]
    ori_obs = []
    tra_obs = []
    ref_filename = os.path.join(args.load_path, '%s_reference_results_%s_tense.txt'%(args.data_name,args.perturb_type))
    transfer_filename = os.path.join(os.path.join(args.load_path, '%s_%s_results_%s_tense.txt'%(args.data_name,args.evaluate_type,args.perturb_type)))
    ref_f = open(ref_filename, "w")
    neutral_num=0
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
            var_id = [neg_idx if label else pos_idx for label in labels]
            text, _ = test_data._to_tensor(
                test_sent[idx:_idx], batch_first=False, device=device)
            feat = torch.tensor(test_feat[idx:_idx], dtype=torch.float, requires_grad=False, device=device)
            domid = torch.tensor(test_domid[idx:_idx], dtype=torch.long, requires_grad=False, device=device)
            # u_embed = model.vae.u_embedding(domid)
            z1, _ = model.vae.lstm_encoder(text[:min(text.shape[0], 10)],u=None)
            ori_z2, _ = model.vae.mlp_encoder(feat,domid)#[N,dim]
            p = model.vae.get_var_prob(ori_z2)#[n,n_vars]
            #if the intervened sentiment is different from the original ones.
            var_list = []
            ori_senti_idx = torch.argmax(p,-1)
            for senti_idx in ori_senti_idx:
                if senti_idx.item() == neg_idx:
                    var_list.append(1) #must be different
                elif senti_idx == pos_idx:
                    var_list.append(2)
                else:
                    var_list.append(0)
                    neutral_num += 1
                    
            if args.evaluate_type == 'transfer':
                if args.perturb_type == 'flip':
                    tra_z2= torch.index_select(model.vae.mlp_encoder.var_embedding, 0, torch.tensor(var_list).cuda())
                elif args.perturb_type == 'shift':
                    pos_emb = style_shift(data=dev_data,pos_sample=pos_inputids,pos_feat=pos_feat_inputs,model=model,domid=int(test_domids[0]),device=device)
                    neg_emb = style_shift(data=dev_data,pos_sample=neg_inputids,pos_feat=neg_feat_inputs,model=model,domid=int(test_domids[0]),device=device)
                    shift_emb = torch.cat([neg_emb if label else pos_emb for label in labels],dim=0)
                    tra_z2 = ori_z2
                    tra_z1 = args.shift_k*shift_emb
            else:
                tra_z2 = ori_z2
            texts = model.vae.decoder.beam_search_decode(z1, tra_z2)
            assert len(texts) == len(test_sent[idx:_idx])
            for label_id,text in enumerate(texts):
                if args.evaluate_type == "transfer":
                    f.write("%d\t%s\n" % (1 - labels[label_id], " ".join(text)))
                else:
                    f.write("%d\t%s\n" % (labels[label_id], " ".join(text)))
                    
            for label_id,sample in enumerate(test_sent[idx:_idx]):
                ref_f.write("%d\t%s\n" % (labels[label_id], " ".join([test_data.vocab.id2word_[word_id] for word_id in sample])))


            idx = _idx
            step += 1

            if step % 100 == 0:
                print(step, idx)

    print("Total %d neutral Unchanged sentences"%neutral_num)
    print(ref_filename,"\n",transfer_filename)
    
def add_args(parser):
    parser.add_argument('--data_name', type=str, default='yelp')
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
    parser.add_argument('--shift_k',type=int,help='shift n sigma style emb', default=1)
    parser.add_argument('--wdomid', type=int, default=0, help = 'train with label or not')
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_args(parser)
    args = parser.parse_args()

    main(args)