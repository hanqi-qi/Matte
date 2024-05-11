
import torch
from utils.exp_utils import create_exp_dir
import config
from models.decomposed_vae import DecomposedVAE
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import argparse
from utils.text_utils import MonoTextData
import numpy as np
import os
from os import system
from finetune_Enc2Classify import evaluate_func,evaluate_func_combinefeat

max_iters = 20

def perturb_text(z1,label,feat,domid,encoder,classifier,device):
    optimizer = optim.Adam(
    [
        {"params": encoder.vae.parameters(), "lr": 5e-4},
        {"params": classifier.parameters(),"lr":1e-10},
    ],1e-10)
    best_acc = 0
    for iter in range(max_iters):
        # z1.requires_grad = False
        # feat.requires_grad = True
        # print(z1[:3,:3])
        
        # combine_feat = feat
        # encode_var_batch,_ = encoder.vae.mlp_encoder(feat)
        # combine_feat = torch.cat([z1,encode_var_batch],axis=-1) #[80]
        # hidden_repr = classifier.lstm_adaptor(encoder,z1,domid,batch_first=False)
        hidden_repr = classifier.mlp_adaptor(encoder,z1,feat,domid,batch_first=False)
        logits = classifier(hidden_repr,input_embs=True)
        # logits = classifier.classify_head(combine_feat)
        float_label = torch.tensor(label, dtype=torch.float, requires_grad=False, device=device)
        feat_loss = F.binary_cross_entropy_with_logits(logits, float_label)
        feat_loss.backward(retain_graph=True)
        optimizer.step()
        classifier.eval()
        # feat_acc = evaluate_func(classifier, dev_batch, dev_label,dev_domid,encoder=encoder,device=device)
        feat_acc = evaluate_func(classifier, [z1], [label],[domid],encoder=encoder,eval_feat=[feat],batch_first=False,device=device)
        perturb_feat = torch.index_select(encoder.vae.mlp_encoder.var_embedding, 0, torch.tensor(label).cuda())
        # feat_acc = evaluate_func(classifier,[combine_feat],[label],encoder=encoder)
        # feat_acc = evaluate_func(classifier, [feat], [label], encoder=encoder)
        if feat_acc > best_acc:
            best_acc = feat_acc
            print(feat_loss.item(),feat_acc)
            # print(encoder.vae.mlp_encoder.var_embedding[:3,:5])
            # perturb_feat,_ = encoder.vae.mlp_encoder(feat)
            # perturb_feat = torch.index_select(encoder.vae.mlp_encoder.var_embedding, 0, torch.tensor(label).cuda())
            # print(perturb_feat[:3,:10])
            update_z1, update_z2 = z1,perturb_feat
            # print(feat_loss.item(),feat_acc)
        # classifier.train()
    print("****************")
    return update_z1, update_z2 



