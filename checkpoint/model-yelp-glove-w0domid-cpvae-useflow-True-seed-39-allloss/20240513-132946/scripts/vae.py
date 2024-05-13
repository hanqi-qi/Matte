# Copyright (c) 2020-present, Royal Bank of Canada.
# Copyright (c) 2020-present, Juxian He
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#################################################################################################
# Code is based on the VAE lagging encoder (https://arxiv.org/abs/1901.05534) implementation
# from https://github.com/jxhe/vae-lagging-encoder by Junxian He
#################################################################################################


import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import uniform_initializer, value_initializer, gumbel_softmax
from .base_network import LSTMEncoder, LSTMDecoder, SemMLPEncoder, SemLSTMEncoder

class DecomposedVAE(nn.Module):
    def __init__(self, lstm_ni, lstm_nh, lstm_nz, mlp_ni, mlp_nz,
                 dec_ni, dec_nh, dec_dropout_in, dec_dropout_out,
                 vocab, n_vars, device, text_only,n_domains,flow_type,flow_nlayer,flow_dim):
        super(DecomposedVAE, self).__init__()
        model_init = uniform_initializer(0.01)
        enc_embed_init = uniform_initializer(0.1)
        dec_embed_init = uniform_initializer(0.1)
        self.lstm_encoder = LSTMEncoder(
            lstm_ni, lstm_nh, lstm_nz, len(vocab), model_init, enc_embed_init,n_domains)
        if text_only:
            self.mlp_encoder = SemLSTMEncoder(
                lstm_ni, lstm_nh, mlp_nz, len(vocab), n_vars, model_init, enc_embed_init, n_domains,device)
        else:
            self.mlp_encoder = SemMLPEncoder(
                mlp_ni, mlp_nz, n_vars, model_init,n_domains, device)
        self.decoder = LSTMDecoder(
            dec_ni, dec_nh, lstm_nz + mlp_nz, dec_dropout_in, dec_dropout_out, vocab,
            model_init, dec_embed_init, device)
        # self.cls_head = nn.Linear(mlp_nz,n_vars)

    def encode_syntax(self, x,u=None, nsamples=1):
        return self.lstm_encoder.encode(x, u,nsamples)

    def encode_semantic(self, x,u, nsamples=1):
        return self.mlp_encoder.encode(x,u, nsamples)

    def decode(self, x, z):
        return self.decoder(x, z)

    def var_loss(self, pos, neg, neg_samples):
        r, _ = self.mlp_encoder(pos, True)
        pos = self.mlp_encoder.encode_var(r)
        pos_scores = (pos * r).sum(-1)
        pos_scores = pos_scores / torch.norm(r, 2, -1)
        pos_scores = pos_scores / torch.norm(pos, 2, -1)
        neg, _ = self.mlp_encoder(neg)
        neg_scores = (neg * r.repeat(neg_samples, 1)).sum(-1)
        neg_scores = neg_scores / torch.norm(r.repeat(neg_samples, 1), 2, -1)
        neg_scores = neg_scores / torch.norm(neg, 2, -1)
        neg_scores = neg_scores.view(neg_samples, -1)
        pos_scores = pos_scores.unsqueeze(0).repeat(neg_samples, 1)
        raw_loss = torch.clamp(1 - pos_scores + neg_scores, min=0.).mean(0)
        srec_loss = raw_loss.mean()
        reg_loss = self.mlp_encoder.orthogonal_regularizer()
        return srec_loss, reg_loss, raw_loss.sum()

    def get_var_prob(self, inputs):
        _, p = self.mlp_encoder.encode_var(inputs, True)
        return p

    def loss(self, x, feat,u=None,tau=1.0, nsamples=1, no_ic=True):
        #TODO(yhq:) concate x and batch_domid
        z1, mu1, logvar1 = self.encode_syntax(x,u,nsamples)
        KL1 = 0.5 * (mu1.pow(2) + logvar1.exp() - logvar1 - 1).sum(1)
        z2, mu2, logvar2 = self.encode_semantic(feat,u,nsamples)
        KL2 = 0.5 * (mu2.pow(2) + logvar2.exp() - logvar2 - 1).sum(1)
        z = torch.cat([z1, z2], -1)
        outputs,_ = self.decode(x[:-1], z)
        reg_ic = None
        return outputs, KL1, KL2, reg_ic

    def calc_mi_q(self, x, feat,u=None):
        mi1 = self.lstm_encoder.calc_mi(x,u)
        mi2 = self.mlp_encoder.calc_mi(feat,u)
        return mi1, mi2