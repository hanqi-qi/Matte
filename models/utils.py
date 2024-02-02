# Copyright (c) 2020-present, Royal Bank of Canada.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import os
import numpy as np
from utils.text_utils import MonoTextData

domain_dict = {"imdb":0,"yelp_dast":1,"amazon":2,"yahoo":3}
domain_i2d = {"0":"imdb","1": "yelp_dast","2":"amazon","3":"yahoo"}
root_path = "/mnt/Data3/hanqiyan/UDA/real_world/data/"

def log_sum_exp(value, dim=None, keepdim=False):
    if dim is not None:
        m, _ = torch.max(value, dim=dim, keepdim=True)
        value0 = value - m
        if keepdim is False:
            m = m.squeeze(dim)
        return m + torch.log(torch.sum(torch.exp(value0), dim=dim, keepdim=keepdim))
    else:
        m = torch.max(value)
        sum_exp = torch.sum(torch.exp(value - m))
        return m + torch.log(sum_exp)

class uniform_initializer(object):
    def __init__(self, stdv):
        self.stdv = stdv

    def __call__(self, tensor):
        nn.init.uniform_(tensor, -self.stdv, self.stdv)

class value_initializer(object):
    def __init__(self, value):
        self.value = value

    def __call__(self, tensor):
        with torch.no_grad():
            tensor.fill_(0.)
            tensor += self.value

def sample_gumbel(shape, eps=1e-20):
    U = torch.rand(shape, requires_grad=True).cuda()
    return -torch.log(-torch.log(U + eps) + eps)

def gumbel_softmax_sample(logits, temperature):
    y = logits + sample_gumbel(logits.size())
    return F.softmax(y / temperature, dim=-1)

def gumbel_softmax(logits, temperature):
    y = gumbel_softmax_sample(logits, temperature)
    shape = y.size()
    _, ind = y.max(dim=-1)
    y_hard = torch.zeros_like(y).view(-1, shape[-1])
    y_hard.scatter_(1, ind.view(-1, 1), 1)
    y_hard = y_hard.view(*shape)
    y_hard = (y_hard - y).detach() + y
    return y_hard

def select_domain_examples(tar_domid,vocab,device,bsz):
    print("Selecting %d examples from %s domain to do Transfer"%(bsz,domain_i2d[str(tar_domid)]))
    #randomly select 10 samples from the target domain.
    tar_data_pth =  os.path.join(root_path+"%s/"%domain_i2d[str(tar_domid)], "dev_data.txt")
    tar_data = MonoTextData(tar_data_pth,4,vocab=vocab)
    random_ids = random.choices(range(len(tar_data.data)),k=bsz)
    tar_domain_examples = [tar_data.data[idx] for idx in random_ids]
    tar_text, _ = tar_data._to_tensor(
                tar_domain_examples, batch_first=False, device=device)
    return tar_text

def select_exampls(data,dev_feat,labels,tar_label,k,vocab,dom_shift=0,test_domid=None):
    # examples = []
    if dom_shift > 0:
        #select from other domain data
        select_domid = 3-int(test_domid)
        select_dom = domain_i2d[str(select_domid)]
        print("Transfer from %s to %s"%(domain_i2d[str(test_domid)],select_dom))
        print()
        dev_data_pth = os.path.join(root_path+"%s/"%select_dom, "dev_data.txt")
        dev_feat_pth = os.path.join(root_path+"%s/"%select_dom, "dev_glove.npy")
        dev_feat = np.load(dev_feat_pth)
        dev_data = MonoTextData(dev_data_pth,4, vocab=vocab)
        data = dev_data.data
        labels = dev_data.labels
    
    select_ids = []
    input_ids = []
    senti_dict ={"0":"Negative","1":"Positive"}
    senti = senti_dict[tar_label]
    print()
    print("Selecting %s examples:"%senti)
    np.random.seed(8888)
    random_ids = random.choices(range(len(data)),k=30)
    for random_id in random_ids:
        if str(labels[random_id]) == tar_label:
            select_ids.append(random_id)
            sent = [vocab.id2word_[word_id] for word_id in data[random_id]]
            # input_ids.append([word_id for word_id in len(data[random_id]])
            input_ids.append([data[random_id][idx] if len(data[random_id])>idx else 0 for idx in range(15)])
            print(" ".join(sent))
        if len(select_ids) == k:
            break
    select_feats = dev_feat[select_ids]
    return select_ids, select_feats, input_ids

def style_shift(data,pos_sample, pos_feat,model,domid,device):
    text, _ = data._to_tensor(
                pos_sample, batch_first=False, device=device)
    domid = torch.tensor(len(pos_sample)*[domid], dtype=torch.long, requires_grad=False, device=device)
    # u_embed = model.vae.u_.30embedding(domid)
    # z1, _ = model.vae.lstm_encoder(text,u_embed)
    pos_emb, _ = model.vae.mlp_encoder(pos_feat,None)
    # context = model.vae.inject_s2flow(u_embed,z1,z2)
    # pos_noise_emb,_ = model.vae.domain_flow_style(z2,context)
    return torch.mean(pos_emb,dim=0,keepdim=True)  #[1,dim]

def select_style_examples(data,labels,tar_label,k,vocab,dom_shift=False,test_domid=None):
    # examples = []
    if dom_shift == True:
        #select from other domain data
        select_domid = 3-int(test_domid)
        select_dom = domain_i2d[str(select_domid)]
        print("Transfer from %s to %s"%(domain_i2d[str(test_domid)],select_dom))
        print()
        dev_data_pth = os.path.join(root_path+"%s/"%select_dom, "dev_data.txt")
        dev_feat_pth = os.path.join(root_path+"%s/"%select_dom, "dev_glove.npy")
        dev_feat = np.load(dev_feat_pth)
        dev_data = MonoTextData(dev_data_pth,4, vocab=vocab)
        data = dev_data.data
        labels = dev_data.labels
    
    select_ids = []
    senti_dict ={"0":"Negative","1":"Positive"}
    senti = senti_dict[tar_label]
    print()
    print("Selecting %s examples:"%senti)
    np.random.seed(8888)
    random_ids = random.choices(range(len(data)),k=k)
    for random_id in random_ids:
        if str(labels[random_id]) == tar_label:
            select_ids.append(random_id)
            sent = [vocab.id2word_[word_id] for word_id in data[random_id]]
            print(" ".join(sent))
        if len(select_ids) == 30:
            break
    if dom_shift == True:
        select_feats = dev_feat[select_ids]
    else:
        select_feats = None
    return select_ids, select_feats


