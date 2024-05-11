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
from utils.text_utils_our import MonoTextData

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


def select_exampls(data,dev_feat,labels,tar_label,k,vocab,dom_shift=0,test_domid=None):
    select_ids = []
    input_ids = []
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
            input_ids.append([word_id for word_id in data[random_id]])
            print(" ".join(sent))
        if len(select_ids) == k:
            break
    select_feats = dev_feat[select_ids]
    return select_ids, select_feats, input_ids

def style_shift(data,pos_sample, pos_feat,model,domid,device):
    text, _ = data._to_tensor(
                pos_sample, batch_first=False, device=device)
    domid = torch.tensor(len(pos_sample)*[domid], dtype=torch.long, requires_grad=False, device=device)
    u_embed = model.vae.u_embedding(domid)
    shift_variable = "z2"
    if shift_variable == "z2":
        pos_emb, _ = model.vae.mlp_encoder(pos_feat,None)
    elif shift_variable == "z1":
        pos_emb, _ = model.vae.lstm_encoder(text,u_embed)
    return torch.mean(pos_emb,dim=0,keepdim=True)  #[1,dim]

def select_style_examples(data,labels,tar_label,k,vocab,dom_shift=False,test_domid=None):
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

    select_feats = None
    return select_ids, select_feats


