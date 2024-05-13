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

from .utils import uniform_initializer, value_initializer, gumbel_softmax
from .base_network import LSTMEncoder, LSTMDecoder, SemMLPEncoder, SemLSTMEncoder
from .flow_network import SigmoidFlow, DenseSigmoidFlow, MLP,DDSF
import torch.nn.functional as F
from .condflow_network import NormalizingCondFlow

class DecomposedVAE(nn.Module): 
    """Add influential Function here"""
    def __init__(self, lstm_ni, lstm_nh, lstm_nz, mlp_ni, mlp_nz,
                 dec_ni, dec_nh, dec_dropout_in, dec_dropout_out,
                 vocab, n_vars, device, text_only,n_domains,flow_type,flow_nlayer,flow_dim,sJacob_rank,u_dim,styleKL,cSparsity,select_k,threshold,start_epoch,vae_pretrain,args):
        super(DecomposedVAE, self).__init__()
        model_init = uniform_initializer(0.01)
        enc_embed_init = uniform_initializer(0.1)
        dec_embed_init = uniform_initializer(0.1)
        self.flow_type = flow_type
        self.c_dim = lstm_nz
        self.s_dim = mlp_nz
        self.u_dim = u_dim
        self.n_vars = n_vars
        self.device = device
        self.normal_distribution = torch.distributions.MultivariateNormal(torch.zeros(self.c_dim+self.s_dim).cuda(), torch.eye(self.c_dim+self.s_dim).cuda())
        self.lstm_encoder = LSTMEncoder(
            lstm_ni, lstm_nh, lstm_nz+mlp_nz, len(vocab), model_init, enc_embed_init,n_domains)
        if text_only:
            self.mlp_encoder = SemLSTMEncoder(
                lstm_ni, lstm_nh, mlp_nz, len(vocab), n_vars, model_init, enc_embed_init, device)
        else:
            self.mlp_encoder = SemMLPEncoder(
                mlp_ni, mlp_nz, n_vars, model_init,n_domains, device)
        self.decoder = LSTMDecoder(
            dec_ni, dec_nh, lstm_nz + mlp_nz, dec_dropout_in, dec_dropout_out, vocab,
            model_init, dec_embed_init, device)
        self.normal_distribution_content = torch.distributions.MultivariateNormal(torch.zeros(lstm_nz).cuda(), torch.eye(lstm_nz).cuda())
        self.normal_distribution_style = torch.distributions.MultivariateNormal(torch.zeros(mlp_nz).cuda(),torch.eye(mlp_nz).cuda())
        self.importance = nn.Parameter(torch.ones((1, self.c_dim)))
        
    
    # flow model for content and style, respectively
        
        self.sJacob_rank = sJacob_rank
        self.styleKL = styleKL
        self.u_embedding = nn.Embedding(4,u_dim)

        

        self.u_embedding = nn.Embedding(4,20)
        self.content_mlp = nn.Linear(self.c_dim+u_dim,self.c_dim)
        self.flow = DDSF(flow_nlayer, 1, flow_dim, 1)
        self.domain_flow_style = NormalizingCondFlow(self.s_dim, self.s_dim,n_layers=1,bound=5, count_bins=8, order='linear')
        self.att_hidden = nn.Linear(u_dim+self.c_dim,1).cuda()
        self.c2ests = nn.Linear(self.c_dim,self.s_dim)
        
        #cls_head for sentiment discriminate
        self.var_linear = nn.Linear(self.s_dim, n_vars)
        self.var_embedding = nn.Parameter(torch.zeros((n_vars, mlp_nz)))
        
        #arguments for c sparsity
        self.select_k = select_k
        self.threshold = threshold
        self.start_epoch = start_epoch
        self.cSparsity = cSparsity
        
        self.domain_flow_content = DDSF(flow_nlayer, 1, flow_dim, 1)
        domain_num_params = self.domain_flow_content.num_params * self.c_dim
        self.domain_mlp_content = MLP(self.u_dim, domain_num_params)
        
    def encode_sent(self, x,u=None, nsamples=1,):
        return self.lstm_encoder.encode(x, u,nsamples)

    def encode_semantic(self, x,u, nsamples=1):
        return self.mlp_encoder.encode(x,u, nsamples)
    
    def decode(self, x, z):
        return self.decoder(x, z)

    def var_loss(self, pos, neg, neg_samples):
        r, _ = self.mlp_encoder(pos, return_origin=True,input_tilde=True) #mean_noise, logvar
        pos = self.mlp_encoder.encode_var(r) #mean->p_mean*var_emb,i.e.,structured_code
        pos_scores = (pos * r).sum(-1) #
        pos_scores = pos_scores / torch.norm(r, 2, -1)
        pos_scores = pos_scores / torch.norm(pos, 2, -1)
        neg, _ = self.mlp_encoder(neg,return_origin=True,input_tilde=True) 
        neg_scores = (neg * r.repeat(neg_samples, 1)).sum(-1)
        neg_scores = neg_scores / torch.norm(r.repeat(neg_samples, 1), 2, -1)
        neg_scores = neg_scores / torch.norm(neg, 2, -1)
        neg_scores = neg_scores.view(neg_samples, -1)
        pos_scores = pos_scores.unsqueeze(0).repeat(neg_samples, 1)
        raw_loss = torch.clamp(1 - pos_scores + neg_scores, min=0.).mean(0)
        srec_loss = raw_loss.mean()
        reg_loss = self.mlp_encoder.orthogonal_regularizer()
        return srec_loss, reg_loss, raw_loss.sum()

    def orthogonal_regularizer(self, norm=100):
        tmp = torch.mm(self.var_embedding, self.var_embedding.permute(1, 0))
        return torch.norm(tmp - norm * torch.diag(torch.ones(self.n_vars, device=self.device)), 2)
    
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
        # zs = zs.squeeze(0)
        tilde_zs,logdet  = self.flow(zs, dsparams)
        return tilde_zs,logdet
    
    def jacobian_loss_function(self, jacobian,epoch):
    # jacobian shape is (latent_dim, batch_size, output_model_shape)
    # where the first batch_size rows correspond to the first latent dimension, etc.
        #print(jacobian.shape)
        latent_dim = jacobian.shape[0]
        batch_size = jacobian.shape[1]
        jacobian = jacobian.reshape((latent_dim, batch_size, -1))
        obs_dim = jacobian.shape[2]
        #apply to the last [sJacob_rank*.s_dim] dimensions
        spare_dim = self.s_dim
        s_loss = torch.sum(torch.abs(jacobian[-spare_dim:,:,:]))/batch_size
        assert len(s_loss.shape)==0, "loss should be a scalar"
        #calculate the constraint for zc
        nInter = 0
        if self.cSparsity>0 and epoch > 0:
            s_jacob = torch.sum(torch.abs(jacobian[-spare_dim:,:,:]),dim=0) #[bs,hidden_x]
            c_jacob = torch.sum(torch.abs(jacobian[:self.c_dim,:,:]),dim=0) #[bs,hidden_x]
            _,topk_sid = torch.topk(s_jacob,k=self.select_k,dim=-1) #[bs,k]
            _,bottomk_cid = torch.topk(c_jacob,self.select_k,largest=False)

            combined = torch.cat((topk_sid,bottomk_cid),dim=1).tolist()
            list_indices = list(map(self.get_frequent_elements,combined))
            flat_indices = torch.tensor([idx+row_idx*c_jacob.shape[1] for row_idx, row in enumerate(list_indices) for idx in row]).to(self.device)
            
            c_loss = torch.sum(torch.index_select(c_jacob.reshape(-1),0,flat_indices)) if len(flat_indices)>0 else torch.tensor([0.],device=self.device)        
            c_loss = torch.sum(c_loss)/batch_size
            return(s_loss,c_loss)
        else:
            return (s_loss,)
    
    def causal_influence(self):
        mask = (self.importance > 0.1).detach().float()
        return mask * self.importance
        
    def domain_content(self, z, u):
        domain_embedding = u  # B,h_dim
        B, _ = domain_embedding.size()
        dsparams = self.domain_mlp_content(domain_embedding)  # B, ndim
        dsparams = dsparams.view(B, self.c_dim, -1)
        #squeeze the 1st dimension for low model
        z = z.squeeze(0)
        tilde_zs, logdet = self.domain_flow_content(z, dsparams)
        return tilde_zs, logdet
    
    def domain_content_concat(self,z,u):
        domain_embedding = u
        B, _ = domain_embedding.size()
        z_u = torch.cat([z,domain_embedding],1)
        z_out = F.leaky_relu(self.content_mlp(z_u),0.1)
        return z_out
    
    def loss(self, x, feat,u=None,tau=1.0, nsamples=1, no_ic=True, epoch=0):
        u_embed = self.u_embedding(u)
        z, mu,logvar = self.encode_sent(x,None,nsamples) #content for reconstruct
        z = z.squeeze(0)
        z1 = z[:,:self.c_dim]
        z2 = z[:,self.c_dim:]
        tilde_z1, logdet_u1 = self.domain_content(z1.unsqueeze(0),u_embed)
        #cmask 
        z1 = self.causal_influence() * z1 # parameterization
        #KL1 for content
        # print(z.shape,z1.shape,z2.shape)
        q_dist_content = torch.distributions.Normal(mu[:,:self.c_dim], torch.exp(torch.clamp(logvar[:,:self.c_dim], min=-10) / 2))
        log_qz_content = q_dist_content.log_prob(z1.squeeze(0))
        log_pz_content = self.normal_distribution_content.log_prob(tilde_z1)
        KL1 = log_qz_content.sum(dim=1)-log_pz_content-logdet_u1
        
        # KL for style 
        context = self.inject_s2flow(u_embed,z2,z1)#[bs,zc_dim]
        # print(context.shape,u_embed.shape,z2.shape)
        tilde_z2,logdet_u2 = self.domain_style(z2,context)
        q_dist_style = torch.distributions.Normal(mu[:,self.c_dim:], torch.exp(torch.clamp(logvar[:,self.c_dim:], min=-10) / 2))
        log_qz_style = q_dist_style.log_prob(z2)
        log_pz_style = self.normal_distribution_style.log_prob(tilde_z2)
        KL2 = log_qz_style.sum(dim=1)-logdet_u2-log_pz_style
   
        z = torch.cat([z1, z2], -1)
        outputs,jaco_matrix = self.decode(x[:-1],z.unsqueeze(0))
        sparse_loss = self.jacobian_loss_function(jaco_matrix,epoch)
        return outputs, KL1, KL2, sparse_loss, torch.mean(torch.abs(self.importance))

    def domain_style(self,z,context):
        context_embedding = context
        B, _ = context_embedding.size()
        dsparams = context.view(B, self.s_dim)
        z = z.squeeze(0)#[B,mlp_dim]
        tilde_zs, logdet = self.domain_flow_style(z, dsparams)
        return tilde_zs, logdet
    
    def calc_mi_q(self, x, feat,u=None):
        u_embed = self.u_embedding(u)
        mi1 = self.lstm_encoder.calc_mi(x,u_embed)
        mi2 = self.mlp_encoder.calc_mi(feat,u_embed)
        return mi1, mi2