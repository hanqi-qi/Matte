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
from collections import OrderedDict
import math
from .utils import log_sum_exp
import numpy as np
from scipy.stats import ortho_group

class BeamSearchNode(object):
    def __init__(self, hiddenstate, previousNode, wordId, logProb, length):
        self.h = hiddenstate
        self.prevNode = previousNode
        self.wordid = wordId
        self.logp = logProb
        self.leng = length

    def eval(self, alpha=1.0):
        reward = 0
        return self.logp / float(self.leng - 1 + 1e-6) + alpha * reward

class MLP(nn.Module):
    def __init__(self, input_dims, n_hiddens, n_class, dropout):
        super(MLP, self).__init__()
        assert isinstance(input_dims, int), 'Invalid type for input_dims!'
        self.input_dims = input_dims
        current_dims = input_dims
        layers = OrderedDict()

        if isinstance(n_hiddens, int):
            n_hiddens = [n_hiddens]
        else:
            n_hiddens = list(n_hiddens)

        for i, n_hidden in enumerate(n_hiddens):
            l_i = i + 1
            layers['fc{}'.format(l_i)] = nn.Linear(current_dims, n_hidden)
            layers['relu{}'.format(l_i)] = nn.ReLU()
            layers['drop{}'.format(l_i)] = nn.Dropout(dropout)
            current_dims = n_hidden
        layers['out'] = nn.Linear(current_dims, n_class)

        self.model = nn.Sequential(layers)

    def forward(self, input):
        return self.model.forward(input)

class GaussianEncoderBase(nn.Module):
    def __init__(self):
        super(GaussianEncoderBase, self).__init__()

    def forward(self, x,):
        raise NotImplementedError

    def sample(self, inputs, nsamples):
        mu, logvar = self.forward(inputs)
        z = self.reparameterize(mu, logvar, nsamples)
        return z, (mu, logvar)

    def encode(self, inputs,u=None, nsamples=1,return_origin=None,input_tilde=None):
        if input_tilde is True:
            tilde_s_prob = self.forward(inputs,u,return_origin=return_origin,input_tilde=True)
            return tilde_s_prob
        else:
            mu, logvar = self.forward(inputs,u,return_origin=return_origin,input_tilde=input_tilde)
            z = self.reparameterize(mu, logvar, nsamples)
            KL = 0.5 * (mu.pow(2) + logvar.exp() - logvar - 1).sum(1)
            return z, mu, logvar

    def reparameterize(self, mu, logvar, nsamples=1):
        batch_size, nz = mu.size()
        std = logvar.mul(0.5).exp()

        mu_expd = mu.unsqueeze(0).expand(nsamples, batch_size, nz)
        std_expd = std.unsqueeze(0).expand(nsamples, batch_size, nz)

        eps = torch.zeros_like(std_expd).normal_()
        return mu_expd + torch.mul(eps, std_expd)

    def sample_from_inference(self, x, nsamples=1):
        mu, logvar = self.forward(x)
        batch_size, nz = mu.size()
        return mu.unsqueeze(0).expand(nsamples, batch_size, nz)

    def eval_inference_dist(self, x, z, param=None):
        nz = z.size(2)
        if not param:
            mu, logvar = self.forward(x)
        else:
            mu, logvar = param

        mu, logvar = mu.unsqueeze(0), logvar.unsqueeze(0)
        var = logvar.exp()
        dev = z - mu

        log_density = -0.5 * ((dev ** 2) / var).sum(dim=-1) - \
            0.5 * (nz * math.log(2 * math.pi) + logvar.sum(-1))

        return log_density.squeeze(0)

    def calc_mi(self, x,u=None):
        mu, logvar = self.forward(x,u)

        x_batch, nz = mu.size()

        neg_entropy = (-0.5 * nz * math.log(2 * math.pi) - 0.5 * (1 + logvar).sum(-1)).mean()

        z_samples = self.reparameterize(mu, logvar, 1)

        mu, logvar = mu.unsqueeze(1), logvar.unsqueeze(1)
        var = logvar.exp()

        dev = z_samples - mu

        log_density = -0.5 * ((dev ** 2) / var).sum(dim=-1) - \
            0.5 * (nz * math.log(2 * math.pi) + logvar.sum(-1))

        log_qz = log_sum_exp(log_density, dim=0) - math.log(x_batch)

        return (neg_entropy - log_qz.mean(-1)).item()

class VAEEncoder(GaussianEncoderBase):
    def __init__(self, ni, nh, nz, vocab_size, model_init, emb_init,n_domains):
        super(VAEEncoder, self).__init__()
        self.embed = nn.Embedding(vocab_size, ni)
        self.n_domains = n_domains
        self.u_embed_layer = nn.Embedding(n_domains,10)

        if n_domains>0:
            ni = ni+10

        self.lstm = nn.LSTM(input_size=ni,
                            hidden_size=nh,
                            num_layers=2,
                            bidirectional=True)
        self.linear = nn.Linear(nh, 2 * nz, bias=False)
        self.reset_parameters(model_init, emb_init)

    def reset_parameters(self, model_init, emb_init):
        for param in self.parameters():
            model_init(param)
        emb_init(self.embed.weight)

    def forward(self, inputs,u=None):
        if len(inputs.size()) > 2:
            word_embed = torch.matmul(inputs, self.embed.weight)
        else:
            word_embed = self.embed(inputs)
            if self.n_domains > 0:
                u_emb = self.u_embed_layer(u)
            # word_embed = self.embed(inputs)#[bs,seq_len,dim]
                word_embed = torch.cat((word_embed,u_emb.unsqueeze(0).repeat(word_embed.shape[0],1,1)),dim=-1)#[]

        outputs, (last_state, last_cell) = self.lstm(word_embed)
        seq_len, bsz, hidden_size = outputs.size()
        hidden_repr = outputs.view(seq_len, bsz, 2, -1).mean(2)
        hidden_repr = torch.max(hidden_repr, 0)[0]

        mean, logvar = self.linear(hidden_repr).chunk(2, -1)
        return mean, logvar

class LSTMEncoder(GaussianEncoderBase):
    def __init__(self, ni, nh, nz, vocab_size, model_init, emb_init,n_domains):
        super(LSTMEncoder, self).__init__()
        self.embed = nn.Embedding(vocab_size, ni)
        self.n_domains = n_domains
        self.u_embed_layer = nn.Embedding(n_domains,10)

        if n_domains>0:
            ni = ni+10

        self.lstm = nn.LSTM(input_size=ni,
                            hidden_size=nh,
                            num_layers=2,
                            bidirectional=True)
        self.linear = nn.Linear(nh, 2 * nz, bias=False)
        self.reset_parameters(model_init, emb_init)

    def reset_parameters(self, model_init, emb_init):
        for param in self.parameters():
            model_init(param)
        emb_init(self.embed.weight)

    def forward(self, inputs,u=None,return_origin=None,input_tilde=None):
        if len(inputs.size()) > 2:
            word_embed = torch.matmul(inputs, self.embed.weight)
        else:
            word_embed = self.embed(inputs)
            if self.n_domains > 0:
                u_emb = self.u_embed_layer(u)
            # word_embed = self.embed(inputs)#[bs,seq_len,dim]
                word_embed = torch.cat((word_embed,u_emb.unsqueeze(0).repeat(word_embed.shape[0],1,1)),dim=-1)#[]

        outputs, (last_state, last_cell) = self.lstm(word_embed)
        seq_len, bsz, hidden_size = outputs.size()
        hidden_repr = outputs.view(seq_len, bsz, 2, -1).mean(2)
        hidden_repr = torch.max(hidden_repr, 0)[0]

        mean, logvar = self.linear(hidden_repr).chunk(2, -1)
        return mean, logvar

class SemLSTMEncoder(GaussianEncoderBase):
    def __init__(self, ni, nh, nz, vocab_size, n_vars, model_init, emb_init,n_domains, device):
        super(SemLSTMEncoder, self).__init__()
        self.n_vars = n_vars
        self.device = device
        self.embed = nn.Embedding(vocab_size, ni)

        self.lstm = nn.LSTM(input_size=ni,
                            hidden_size=nh,
                            num_layers=1,
                            bidirectional=True)
        self.linear = nn.Linear(nh, 2 * nz, bias=False)
        self.var_embedding = nn.Parameter(torch.zeros((n_vars, nz)))
        self.var_linear = nn.Linear(nz, n_vars)
        self.reset_parameters(model_init, emb_init)

    def reset_parameters(self, model_init, emb_init):
        for param in self.parameters():
            model_init(param)
        emb_init(self.embed.weight)

    def encode_var(self, inputs, return_p=False):
        logits = self.var_linear(inputs)
        prob = F.softmax(logits, -1)
        if return_p:
            return torch.matmul(prob, self.var_embedding), prob
        return torch.matmul(prob, self.var_embedding)

    def orthogonal_regularizer(self, norm=10):
        tmp = torch.mm(self.var_embedding, self.var_embedding.permute(1, 0))
        return torch.norm(tmp - norm * torch.diag(torch.ones(self.n_vars, device=self.device)), 2)

    def forward(self, inputs,u=None, return_origin=False):
        if len(inputs.size()) > 2:
            word_embed = torch.matmul(inputs, self.embed.weight)
        else:
            word_embed = self.embed(inputs)

        outputs, (last_state, last_cell) = self.lstm(word_embed)
        seq_len, bsz, hidden_size = outputs.size()
        hidden_repr = outputs.view(seq_len, bsz, 2, -1).mean(2)
        hidden_repr = torch.max(hidden_repr, 0)[0]

        mean, logvar = self.linear(hidden_repr).chunk(2, -1)
        if return_origin:
            return mean, logvar
        return self.encode_var(mean), logvar

class SemMLPEncoder(GaussianEncoderBase):
    def __init__(self, ni, nz, n_vars, model_init,n_domains, device):
        super(SemMLPEncoder, self).__init__()
        self.n_vars = n_vars
        self.device = device

        self.output = nn.Linear(ni, 2 * nz)
        self.var_embedding = nn.Parameter(torch.zeros((n_vars, nz)))

        self.var_linear = nn.Linear(nz, n_vars)
        self.reset_parameters(model_init)

    def reset_parameters(self, model_init):
        for param in self.parameters():
            model_init(param)

    def encode_var(self, inputs, return_p=False):
        logits = self.var_linear(inputs)
        prob = F.softmax(logits, -1)
        if return_p:
            return torch.matmul(prob, self.var_embedding), prob
        return torch.matmul(prob, self.var_embedding)

    def orthogonal_regularizer(self, norm=100):
        tmp = torch.mm(self.var_embedding, self.var_embedding.permute(1, 0))
        return torch.norm(tmp - norm * torch.diag(torch.ones(self.n_vars, device=self.device)), 2)

    def forward(self, inputs,u=None, return_origin=False, input_tilde=False):
        if input_tilde: #map the tilde_s to n_var axis for
            return self.encode_var(inputs,return_p=True)
        
        if return_origin is False:
            mean, logvar = self.output(inputs).chunk(2, -1)
            return self.encode_var(mean), logvar
        else:
            mean, logvar = self.output(inputs).chunk(2, -1)
            return mean, logvar

class LSTMDecoder(nn.Module):
    def __init__(self, ni, nh, nz, dropout_in, dropout_out, vocab,
                 model_init, emb_init, device):
        super(LSTMDecoder, self).__init__()
        self.nz = nz
        self.vocab = vocab
        self.device = device

        self.embed = nn.Embedding(len(vocab), ni, padding_idx=-1)

        self.dropout_in = nn.Dropout(dropout_in)
        self.dropout_out = nn.Dropout(dropout_out)

        self.trans_linear = nn.Linear(nz, nh, bias=False)

        self.lstm = nn.LSTM(input_size=ni + nz,
                            hidden_size=nh,
                            num_layers=1)

        self.pred_linear = nn.Linear(nh, len(vocab), bias=False)

        self.reset_parameters(model_init, emb_init)

    def reset_parameters(self, model_init, emb_init):
        for param in self.parameters():
            model_init(param)
        emb_init(self.embed.weight)
        
    def decoder_predict(self,z,word_embed):
        n_sample, batch_size,_ = z.size()
        seq_len = word_embed.size(0)
        if n_sample == 1:
            z_ = z.expand(seq_len, batch_size, self.nz)
        else:
            raise NotImplementedError
        word_embed = torch.cat((word_embed, z_), -1)
        z = z.reshape(batch_size * n_sample, self.nz)
        c_init = self.trans_linear(z).unsqueeze(0)
        h_init = torch.tanh(c_init)
        output, _ = self.lstm(word_embed, (h_init, c_init))

        # output_logits = self.pred_linear(output)

        return output
    


    def jacobian_loss_function(self, jacobian):
    # jacobian shape is (latent_dim, batch_size, output_model_shape)
    # where the first batch_size rows correspond to the first latent dimension, etc.
        #print(jacobian.shape)
        latent_dim = jacobian.shape[0]
        batch_size = jacobian.shape[1]
        jacobian = jacobian.reshape((latent_dim, batch_size, -1))
        obs_dim = jacobian.shape[2]
        # loss = torch.sum(torch.abs(jacobian))/batch_size
        loss = torch.sum(torch.abs(jacobian[:self.c_dim,:,:self.c_dim]))/batch_size
        assert len(loss.shape)==0, "loss should be a scalar"
        return(loss)
    
    def compute_generator_jacobian_optimized(self,embedding,word_embed,epsilon_scale = 0.001):
        if embedding.ndim == 3:
            n_sample = embedding.shape[0] #1
            batch_size = embedding.shape[1]
            latent_dim = embedding.shape[2]
        else:
            embedding = embedding.unsqueeze(0)
        
        embedding = embedding.squeeze(0)

        encoding_rep = embedding.repeat(latent_dim + 1,1)
        other_rep = word_embed.repeat(1,latent_dim + 1,1).detach().clone()

        delta = torch.eye(latent_dim)\
                    .reshape(latent_dim, 1, latent_dim)\
                    .repeat(1, batch_size, 1)\
                    .reshape(latent_dim*batch_size, latent_dim)
        delta = torch.cat((delta, torch.zeros(batch_size,latent_dim))).cuda()

        epsilon = torch.tensor(epsilon_scale).cuda()     
        encoding_rep += epsilon * delta
        recons = self.decoder_predict(encoding_rep.unsqueeze(0),other_rep)
        # recons = model._decode(encoding_rep)
        temp_calc_shape = [other_rep.shape[0]]+ [latent_dim+1,batch_size] + list(recons.shape[2:])
        recons = recons.reshape(temp_calc_shape)
        recons = torch.sum((recons[:,:-1,:,:] - recons[:,-1,:,:].unsqueeze(1))/epsilon,dim=0) #[seq_len,dim,bs,out_dim]
        return(recons) #[in_dim,bs,out_dim]
    
    def forward(self, inputs, z): #decoder with Jacobian Calculation
        n_sample, batch_size, _ = z.size()
        seq_len = inputs.size(0)

        word_embed = self.embed(inputs)
        word_embed = self.dropout_in(word_embed)
        jacobian_matrix = self.compute_generator_jacobian_optimized(z,word_embed)
        output = self.decoder_predict(z,word_embed)
        output_logits = self.pred_linear(output)
        # z = z.view(batch_size * n_sample, self.nz)
        # c_init = self.trans_linear(z).unsqueeze(0)
        # h_init = torch.tanh(c_init)
        # output, _ = self.lstm(word_embed, (h_init, c_init))
        # output = self.dropout_out(output)
        # output_logits = self.pred_linear(output)
        return output_logits.view(-1, batch_size, len(self.vocab)),jacobian_matrix
    

    # def forward(self, inputs, z):
    #     n_sample, batch_size, _ = z.size()
    #     seq_len = inputs.size(0)

    #     word_embed = self.embed(inputs)
    #     word_embed = self.dropout_in(word_embed)

    #     if n_sample == 1:
    #         z_ = z.expand(seq_len, batch_size, self.nz)
    #     else:
    #         raise NotImplementedError

    #     word_embed = torch.cat((word_embed, z_), -1)

    #     z = z.view(batch_size * n_sample, self.nz)
    #     c_init = self.trans_linear(z).unsqueeze(0)
    #     h_init = torch.tanh(c_init)
    #     output, _ = self.lstm(word_embed, (h_init, c_init))

    #     output = self.dropout_out(output)
    #     output_logits = self.pred_linear(output)

    #     return output_logits.view(-1, batch_size, len(self.vocab))

    def decode(self, z, greedy=True):
        batch_size = z.size(0)
        decoded_batch = [[] for _ in range(batch_size)]

        c_init = self.trans_linear(z).unsqueeze(0)
        h_init = torch.tanh(c_init)

        decoder_hidden = (h_init, c_init)
        decoder_input = torch.tensor([self.vocab["<s>"]] * batch_size, dtype=torch.long,
                                     device=self.device).unsqueeze(0)
        end_symbol = torch.tensor([self.vocab["</s>"]] * batch_size, dtype=torch.long,
                                  device=self.device)

        mask = torch.ones((batch_size), dtype=torch.uint8, device=self.device)
        length_c = 1
        while mask.sum().item() != 0 and length_c < 100:
            word_embed = self.embed(decoder_input)
            word_embed = torch.cat((word_embed, z.unsqueeze(0)), dim=-1)

            output, decoder_hidden = self.lstm(word_embed, decoder_hidden)

            decoder_output = self.pred_linear(output)
            output_logits = decoder_output.squeeze(0)

            if greedy:
                select_index = torch.argmax(output_logits, dim=1)
            else:
                sample_prob = F.softmax(output_logits, dim=1)
                select_index = torch.multinomial(sample_prob, num_samples=1).squeeze(1)

            decoder_input = select_index.unsqueeze(0)
            length_c += 1

            for i in range(batch_size):
                if mask[i].item():
                    decoded_batch[i].append(self.vocab.id2word(select_index[i].item()))

            mask = torch.mul((select_index != end_symbol), mask)

        return decoded_batch

    def beam_search_decode(self, z1, z2=None, K=5, max_t=20):
        decoded_batch = []
        if z2 is not None:
            z = torch.cat([z1, z2], -1)
        else:
            z = z1
        batch_size, nz = z.size()

        c_init = self.trans_linear(z).unsqueeze(0)
        h_init = torch.tanh(c_init)

        for idx in range(batch_size):
            decoder_input = torch.tensor([[self.vocab["<s>"]]], dtype=torch.long,
                                         device=self.device)
            decoder_hidden = (h_init[:, idx, :].unsqueeze(1), c_init[:, idx, :].unsqueeze(1))
            node = BeamSearchNode(decoder_hidden, None, decoder_input, 0.1, 1)
            live_hypotheses = [node]

            completed_hypotheses = []

            t = 0
            while len(completed_hypotheses) < K and t < max_t:
                t += 1

                decoder_input = torch.cat([node.wordid for node in live_hypotheses], dim=1)

                decoder_hidden_h = torch.cat([node.h[0] for node in live_hypotheses], dim=1)
                decoder_hidden_c = torch.cat([node.h[1] for node in live_hypotheses], dim=1)

                decoder_hidden = (decoder_hidden_h, decoder_hidden_c)

                word_embed = self.embed(decoder_input)
                word_embed = torch.cat((word_embed, z[idx].view(1, 1, -1).expand(
                    1, len(live_hypotheses), nz)), dim=-1)

                output, decoder_hidden = self.lstm(word_embed, decoder_hidden)

                output_logits = self.pred_linear(output)
                decoder_output = F.log_softmax(output_logits, dim=-1)

                prev_logp = torch.tensor([node.logp for node in live_hypotheses],
                                         dtype=torch.float, device=self.device)
                decoder_output = decoder_output + prev_logp.view(1, len(live_hypotheses), 1)

                decoder_output = decoder_output.view(-1)

                log_prob, indexes = torch.topk(decoder_output, K - len(completed_hypotheses))

                live_ids = indexes // len(self.vocab)
                word_ids = indexes % len(self.vocab)

                live_hypotheses_new = []
                for live_id, word_id, log_prob_ in zip(live_ids, word_ids, log_prob):
                    node = BeamSearchNode((
                        decoder_hidden[0][:, live_id, :].unsqueeze(1),
                        decoder_hidden[1][:, live_id, :].unsqueeze(1)),
                        live_hypotheses[live_id], word_id.view(1, 1), log_prob_, t)

                    if word_id.item() == self.vocab["</s>"]:
                        completed_hypotheses.append(node)
                    else:
                        live_hypotheses_new.append(node)

                live_hypotheses = live_hypotheses_new

                if len(completed_hypotheses) == K:
                    break

            for live in live_hypotheses:
                completed_hypotheses.append(live)

            utterances = []
            for n in sorted(completed_hypotheses, key=lambda node: node.logp, reverse=True):
                utterance = []
                utterance.append(self.vocab.id2word(n.wordid.item()))
                while n.prevNode is not None:
                    n = n.prevNode
                    utterance.append(self.vocab.id2word(n.wordid.item()))

                utterance = utterance[::-1]
                utterances.append(utterance)

                break

            decoded_batch.append(utterances[0])

        return decoded_batch
