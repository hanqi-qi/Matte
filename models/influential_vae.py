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
import collections

from .utils import uniform_initializer, value_initializer, gumbel_softmax
from .base_network import LSTMEncoder, LSTMDecoder, SemMLPEncoder, SemLSTMEncoder
from .flow_network import SigmoidFlow, DenseSigmoidFlow, MLP,DDSF
import torch.nn.functional as F
from .condflow_network import NormalizingCondFlow

class DecomposedVAE(nn.Module): 
    """Add influential Function here"""
    def __init__(self, lstm_ni, lstm_nh, lstm_nz, mlp_ni, mlp_nz,
                 dec_ni, dec_nh, dec_dropout_in, dec_dropout_out,
                 vocab, n_vars, device, text_only,n_domains,flow_type,flow_nlayer,flow_dim):
        super(DecomposedVAE, self).__init__()
        model_init = uniform_initializer(0.01)
        enc_embed_init = uniform_initializer(0.1)
        dec_embed_init = uniform_initializer(0.1)
        self.flow_type = flow_type
        self.c_dim = lstm_nz
        self.s_dim = mlp_nz
        self.select_k = 20
        self.device = device
        self.importance = nn.Parameter(torch.ones((1, self.c_dim)))
        self.normal_distribution = torch.distributions.MultivariateNormal(torch.zeros(self.c_dim+self.s_dim).cuda(), torch.eye(self.c_dim+self.s_dim).cuda())

        self.lstm_encoder = LSTMEncoder(
                lstm_ni, lstm_nh, lstm_nz, len(vocab), model_init, enc_embed_init,n_domains)
        self.mlp_encoder = SemMLPEncoder(
                mlp_ni, mlp_nz, n_vars, model_init,n_domains, device)
        self.decoder = LSTMDecoder(
            dec_ni, dec_nh, lstm_nz + mlp_nz, dec_dropout_in, dec_dropout_out, vocab,
            model_init, dec_embed_init, device)
        self.normal_distribution_content = torch.distributions.MultivariateNormal(torch.zeros(lstm_nz).cuda(), torch.eye(lstm_nz).cuda())
        self.normal_distribution_style = torch.distributions.MultivariateNormal(torch.zeros(mlp_nz).cuda(),torch.eye(mlp_nz).cuda())
    
    # flow model for content and style, respectively
        self.u_embedding = nn.Embedding(4,20)
        self.domain_flow = SigmoidFlow(flow_dim)
        self.content_mlp = nn.Linear(self.c_dim+20,self.c_dim)
        self.flow = DDSF(flow_nlayer, 1, flow_dim, 1)
        self.domain_flow_style = NormalizingCondFlow(self.s_dim, self.s_dim,n_layers=1,bound=5, count_bins=8, order='linear')
        self.att_hidden = nn.Linear(20+self.c_dim,1).cuda()
        self.c2ests = nn.Linear(self.c_dim,self.s_dim)

    def encode_syntax(self, x,u=None, nsamples=1,):
        return self.lstm_encoder.encode(x, u,nsamples)#get mu, logvar

    def encode_semantic(self, x,u, nsamples=1,return_origin=None,input_tilde=None):
        return self.mlp_encoder.encode(x,u, nsamples,return_origin=return_origin,input_tilde=input_tilde)

    def decode(self, x, z):
        return self.decoder(x, z)
    
    def causal_influence(self):
        mask = (self.importance > 0.1).detach().float()
        return mask * self.importance

    def get_frequent_elements(self,element_list,lb=2):
        counter = collections.Counter(element_list)
        return [element for element, count in counter.items() if count >= lb]
    
    def var_loss(self, pos, neg, neg_samples):
        #unsupervised loss
        r, _ = self.mlp_encoder(pos, True) #[bs,hz]
        pos = self.mlp_encoder.encode_var(r) #[bs,hz]
        pos_scores = (pos * r).sum(-1) #[bs]
        pos_scores = pos_scores / torch.norm(r, 2, -1)
        pos_scores = pos_scores / torch.norm(pos, 2, -1)
        neg, _ = self.mlp_encoder(neg)
        neg_scores = (neg * r.repeat(neg_samples, 1)).sum(-1)
        neg_scores = neg_scores / torch.norm(r.repeat(neg_samples, 1), 2, -1)
        neg_scores = neg_scores / torch.norm(neg, 2, -1)
        neg_scores = neg_scores.view(neg_samples, -1)
        pos_scores = pos_scores.unsqueeze(0).repeat(neg_samples, 1)
        raw_loss = torch.clamp(1 - pos_scores + neg_scores, min=0.).mean(0)#loss from cpvae
        srec_loss = raw_loss.mean()
        reg_loss = self.mlp_encoder.orthogonal_regularizer() #loss from cpvae
        return srec_loss, reg_loss, raw_loss.sum()

    def get_var_prob(self, inputs):
        _, p = self.mlp_encoder.encode_var(inputs, True)
        return p
    
    def inject_s2flow(self,u,zs,zc):
        content_unit = torch.cat((zc.squeeze(0),u),1)
        att = torch.tanh(self.att_hidden(content_unit))
        context = zc.squeeze(0) * att
        esti_s = self.c2ests(context)
        return esti_s
    
    def domain_influence(self, zs, u):
        domain_embedding = self.u_embedding(u)  # B,h_dim
        B, _ = domain_embedding.size()
        dsparams = self.domain_mlp_style(domain_embedding)  # B, ndim
        dsparams = dsparams.view(B, self.s_dim, -1)
        tilde_zs,logdet  = self.flow(zs, dsparams)
        return tilde_zs,logdet
    
    def jacobian_loss_function(self, jacobian):
        latent_dim = jacobian.shape[0]
        batch_size = jacobian.shape[1]
        jacobian = jacobian.reshape((latent_dim, batch_size, -1))
        obs_dim = jacobian.shape[2]
        spare_dim = self.s_dim
        s_loss = torch.sum(torch.abs(jacobian[-spare_dim:,:,:]))/batch_size
        assert len(s_loss.shape)==0, "loss should be a scalar"
        #calculate the constraint for zc
        nInter = 0
        s_jacob = torch.sum(torch.abs(jacobian[-spare_dim:,:,:]),dim=0) #[bs,hidden_x]
        c_jacob = torch.sum(torch.abs(jacobian[:self.c_dim,:,:]),dim=0) #[bs,hidden_x]
        _,topk_sid = torch.topk(s_jacob,k=self.select_k,dim=-1) #[bs,k]
        _,bottomk_cid = torch.topk(c_jacob,self.select_k,largest=False)

        combined = torch.cat((topk_sid,bottomk_cid),dim=1).tolist()
        list_indices = list(map(self.get_frequent_elements,combined))
        flat_indices = torch.tensor([idx+row_idx*c_jacob.shape[1] for row_idx, row in enumerate(list_indices) for idx in row]).to(self.device)
        
        c_loss = torch.sum(torch.index_select(c_jacob.reshape(-1),0,flat_indices)) if len(flat_indices)>0 else torch.tensor([0.],device=self.device)       
        c_loss = torch.sum(c_loss)/batch_size
        #sparsity loss(including s and c) and cmask loss
        return(s_loss,c_loss)


    def loss(self, x, feat,u=None,tau=1.0, nsamples=1, no_ic=True):
        u_embed = self.u_embedding(u)
        z1, mu1,logvar1 = self.encode_syntax(x,None,nsamples) #content for reconstruct
        z1 = self.causal_influence() * z1 #cmask
        z1_u = self.domain_content_concat(z1.squeeze(0),u=u_embed) #noise_c
        KL1 = 0.5 * (mu1.squeeze(0).pow(2) + logvar1.exp() - logvar1 - 1).sum(1)

        z2, mu2,logvar2 = self.encode_semantic(feat,None,nsamples=1,return_origin=True)
        context = self.inject_s2flow(u_embed,z2,z1)
        z2_u,logdet = self.domain_style(z2,context)
        # cls_rep,cls_prob = self.encode_semantic(z2_u,u=None,nsamples=1,return_origin=False,input_tilde=True)
        KL2 = 0.5 * (mu2.squeeze(0).pow(2) + logvar2.exp() - logvar2 - 1).sum(1)
        
        #combine c and s
        z = torch.cat([z1, z2], -1)
        outputs,jaco_matrix = self.decode(x[:-1],z)
        sparse_loss = self.jacobian_loss_function(jaco_matrix)
        cmask_loss = torch.mean(torch.abs(self.importance))
        return outputs, KL1, KL2, sparse_loss,cmask_loss

    def domain_content_concat(self,z,u):
        domain_embedding = u
        B, _ = domain_embedding.size()
        z_u = torch.cat([z,domain_embedding],1)
        z_out = F.leaky_relu(self.content_mlp(z_u),0.1)
        return z_out

    def domain_style(self,z,context):
        context_embedding = context
        B, _ = context_embedding.size()
        # dsparams = self.domain_mlp_style(context_embedding)  # B, ndim
        dsparams = context.view(B, self.s_dim)
        z = z.squeeze(0)#[B,mlp_dim]
        tilde_zs, logdet = self.domain_flow_style(z, dsparams)
        return tilde_zs, logdet

    def calc_mi_q(self, x, feat,u=None):
        u_embed = self.u_embedding(u)
        mi1 = self.lstm_encoder.calc_mi(x,u_embed)
        mi2 = self.mlp_encoder.calc_mi(feat,u_embed)
        return mi1, mi2
    
    # def cuda(self):
    #     self.encoder.cuda()
    #     self.decoder.cuda()