# Copyright (c) 2020-present, Royal Bank of Canada.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from .vae import DecomposedVAE as VAE
from .influential_vae import DecomposedVAE as HiVAE
from .pretrainedvae import VAE as PretrainVAE
import numpy as np
import time
import os

class DecomposedVAE:
    def __init__(self, train, valid, test, feat, bsz, save_path, logging, log_interval, num_epochs,
                 enc_lr, dec_lr, warm_up, kl_start, beta1, beta2, srec_weight, reg_weight,
                 aggressive, text_only,n_domains,train_schema,wdomid,sSparsity,lambda_mask,vae_params):
        super(DecomposedVAE, self).__init__()
        self.bsz = bsz
        self.save_path = save_path
        self.logging = logging
        self.log_interval = log_interval
        self.num_epochs = num_epochs
        self.enc_lr = enc_lr
        self.dec_lr = dec_lr
        self.warm_up = warm_up
        self.kl_weight = kl_start
        self.beta1 = beta1
        self.beta2 = beta2
        self.lambda_mask = lambda_mask
        self.srec_weight = srec_weight
        self.reg_weight = reg_weight
        self.aggressive = aggressive
        self.opt_dict = {"not_improved": 0, "lr": 1., "best_loss": 1e4}
        self.pre_mi = 0
        self.use_cuda = torch.cuda.is_available()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.text_only = text_only
        self.n_domains = n_domains
        self.train_schema = train_schema
        self.wdomid = wdomid
        self.sSparsity = sSparsity
        
        if self.text_only:
            self.train_data, self.train_feat, self.train_label = train, train, train
            self.valid_data, self.valid_feat, self.valid_label = valid, valid
            self.test_data, self.test_feat, self.test_label = test, test
        else:
            self.train_data, self.train_feat, self.train_domid,_  = train
            self.valid_data, self.valid_feat, self.valid_domid,_ = valid
            self.test_data, self.test_feat,self.test_domid,_ = test
        self.feat = feat
        if vae_params["flow_type"] == "True":
            print("Train Our Model")
            self.vae = HiVAE(**vae_params)
        elif vae_params["flow_type"] == "False":
            print("Train Backbone")
            self.vae = VAE(**vae_params)
        if self.use_cuda:
            self.vae.cuda()

        if vae_params["vae_pretrain"] == 0:
            self.enc_params = list(self.vae.lstm_encoder.parameters()) + \
            list(self.vae.mlp_encoder.parameters())
            self.enc_optimizer = optim.Adam(self.enc_params, lr=self.enc_lr)
            self.dec_optimizer = optim.SGD(self.vae.decoder.parameters(), lr=self.dec_lr)
            self.flow_params = []
            for name,params in self.vae.named_parameters():
                # print(name)
                if ("encoder" not in name) or ("decoder" in name):
                    self.flow_params.append(params)
            self.flow_optimizer = optim.Adam(self.flow_params, lr=1e-4)
        
        self.nbatch = len(self.train_data)
        self.anneal_rate = (1.0 - kl_start) / (warm_up * self.nbatch)

    def train(self, epoch):
        self.vae.train()

        total_rec_loss = 0
        total_kl1_loss = 0
        total_kl2_loss = 0
        total_srec_loss = 0
        total_sparse_loss = 0
        start_time = time.time()
        step = 0
        num_words = 0
        num_sents = 0
        wContentSparsity = 0
        shuffle_nbatch = np.random.permutation(range(self.nbatch))

        for idx in shuffle_nbatch:
            batch_data = self.train_data[idx]
            batch_feat = self.train_feat[idx]
            # batch_label = self.train_label[idx]
            sent_len, batch_size = batch_data.size()
            # if self.n_domains=1:
            batch_domid = self.train_domid[idx]
            shift = np.random.randint(max(1, sent_len - 9))
            batch_data = batch_data[shift:min(sent_len, shift + 10), :]
            sent_len, batch_size = batch_data.size()

            target = batch_data[1:]
            num_words += (sent_len - 1) * batch_size
            num_sents += batch_size
            self.kl_weight = min(1.0, self.kl_weight + self.anneal_rate)
            beta1 = self.beta1 if self.beta1 else self.kl_weight
            beta2 = self.beta2 if self.beta2 else self.kl_weight

            loss = 0

            self.enc_optimizer.zero_grad()
            self.dec_optimizer.zero_grad()
            self.flow_optimizer.zero_grad()

            vae_logits, vae_kl1_loss, vae_kl2_loss, sparse_loss,cmask_loss = self.vae.loss(
                x=batch_data, feat=batch_feat,u=batch_domid, no_ic=self.sSparsity == 0)
            vae_logits = vae_logits.view(-1, vae_logits.size(2))
            vae_rec_loss = F.cross_entropy(vae_logits, target.view(-1), reduction="none")
            vae_rec_loss = vae_rec_loss.view(-1, batch_size).sum(0)

            vae_loss = vae_rec_loss + beta1 * vae_kl1_loss + beta2 * vae_kl2_loss
                
            if self.sSparsity > 0:
                if len(sparse_loss)==1:
                    vae_loss += self.sSparsity*sparse_loss[0]
                    sparse_loss = sparse_loss[0]
                else:
                    vae_loss = vae_loss + self.sSparsity*sparse_loss[0] + self.sSparsity*sparse_loss[1]
                    sparse_loss = sparse_loss[0]+sparse_loss[1]
                    wContentSparsity = 1
            vae_loss = vae_loss.mean()
            total_rec_loss += vae_rec_loss.sum().item()
            total_kl1_loss += vae_kl1_loss.sum().item()
            total_kl2_loss += vae_kl2_loss.sum().item()
            total_sparse_loss += sparse_loss.item()
            
            loss = loss + vae_loss+self.lambda_mask*cmask_loss
                
            loss.backward()

            nn.utils.clip_grad_norm_(self.vae.parameters(), 5.0)

            self.enc_optimizer.step()
            self.dec_optimizer.step()
            # if epoch < 2:
            self.flow_optimizer.step()

            if step % self.log_interval == 0 and step > 0:
                cur_rec_loss = total_rec_loss / num_sents
                cur_kl1_loss = total_kl1_loss / num_sents
                cur_kl2_loss = total_kl2_loss / num_sents
                # cur_vae_loss = cur_rec_loss
                cur_vae_loss = cur_rec_loss + cur_kl1_loss + cur_kl2_loss
                cur_srec_loss = total_srec_loss / num_sents
                cur_sparse_loss = total_sparse_loss / num_sents
                elapsed = time.time() - start_time
                self.logging(f'wContentSparsity:{wContentSparsity}')
                self.logging(
                    '| epoch {:2d} | {:5d}/{:5d} batches | {:5.2f} ms/batch | loss {:3.2f} | '
                    'recon {:3.2f} | sparse {:5.4f}| kl1 {:3.2f} | kl2 {:3.2f} | srec {:3.2f}'.format(
                        epoch, step, self.nbatch, elapsed * 1000 / self.log_interval, cur_vae_loss,
                        cur_rec_loss, cur_sparse_loss,cur_kl1_loss, cur_kl2_loss, cur_srec_loss))
                total_rec_loss = 0
                total_kl1_loss = 0
                total_kl2_loss = 0
                total_srec_loss = 0
                total_sparse_loss = 0
                num_sents = 0
                num_words = 0
                acc_list = []
                start_time = time.time()
            step += 1
            torch.cuda.empty_cache()
            
    def evaluate_our(self, eval_data, eval_feat,eval_domid):
        self.vae.eval()

        total_rec_loss = 0
        total_kl1_loss = 0
        total_kl2_loss = 0
        total_cls_loss = 0
        total_mi1 = 0
        total_mi2 = 0
        num_sents = 0
        num_words = 0

        with torch.no_grad():
            for batch_data, batch_feat,batch_domid in zip(eval_data, eval_feat,eval_domid):
                sent_len, batch_size = batch_data.size()
                shift = np.random.randint(max(1, sent_len - 9))
                batch_data = batch_data[shift:min(sent_len, shift + 10), :]
                sent_len, batch_size = batch_data.size()
                target = batch_data[1:]

                num_sents += batch_size
                num_words += (sent_len - 1) * batch_size

                vae_logits, vae_kl1_loss, vae_kl2_loss, _,_= self.vae.loss(
                    batch_data, batch_feat,u=batch_domid)
                vae_logits = vae_logits.view(-1, vae_logits.size(2))
                vae_rec_loss = F.cross_entropy(vae_logits, target.view(-1), reduction="none")
                total_rec_loss += vae_rec_loss.sum().item()
                total_kl1_loss += vae_kl1_loss.sum().item()
                total_kl2_loss += vae_kl2_loss.sum().item()

                mi1, mi2 = self.vae.calc_mi_q(batch_data, batch_feat, u=batch_domid) #only in inference phrase
                total_mi1 += mi1 * batch_size
                total_mi2 += mi2 * batch_size

        cur_rec_loss = total_rec_loss / num_sents 
        cur_cls_loss = 0
        cur_kl1_loss = total_kl1_loss / num_sents
        cur_kl2_loss = total_kl2_loss / num_sents
        cur_vae_loss = cur_rec_loss
        cur_mi1 = total_mi1 / num_sents
        cur_mi2 = total_mi2 / num_sents
        return cur_vae_loss, cur_rec_loss, cur_cls_loss,cur_kl1_loss, cur_kl2_loss, cur_mi1, cur_mi2

    def fit(self):
        best_loss = 1e4
        decay_cnt = 0
        for epoch in range(1, self.num_epochs + 1):
            epoch_start_time = time.time()
            self.train(epoch)
            val_loss = self.evaluate_our(self.valid_data, self.valid_feat,self.valid_domid)

            vae_loss = val_loss[1]

            if self.aggressive:
                cur_mi = val_loss[4]
                self.logging("pre mi: %.4f, cur mi:%.4f" % (self.pre_mi, cur_mi))
                if cur_mi < self.pre_mi:
                    self.aggressive = False
                    self.logging("STOP BURNING")

                self.pre_mi = cur_mi

            if vae_loss < best_loss:
                self.save(self.save_path,epoch)
                best_loss = vae_loss

            if vae_loss > self.opt_dict["best_loss"]:
                self.opt_dict["not_improved"] += 1
                if self.opt_dict["not_improved"] >= 2 and epoch >= 15:
                    self.opt_dict["not_improved"] = 0
                    self.opt_dict["lr"] = self.opt_dict["lr"] * 0.5
                    # self.load(self.save_path,epoch)
                    decay_cnt += 1
                    self.dec_optimizer = optim.SGD(
                        self.vae.decoder.parameters(), lr=self.opt_dict["lr"])
            else:
                self.opt_dict["not_improved"] = 0
                self.opt_dict["best_loss"] = vae_loss

            if decay_cnt == 3:
                break

            self.logging('-' * 75)
            self.logging('| end of epoch {:2d} | time {:5.2f}s | '
                         'kl_weight {:.2f} | vae_lr {:.2f} | loss {:3.2f}'.format(
                             epoch, (time.time() - epoch_start_time),
                             self.kl_weight, self.opt_dict["lr"], val_loss[0]))
            self.logging('| recon {:3.2f} | cls {:3.2f}| kl1 {:3.2f} | kl2 {:3.2f} | '
                         'mi1 {:3.2f} | mi2 {:3.2f}'.format(
                             val_loss[1], val_loss[2], val_loss[3],
                             val_loss[4], val_loss[5], val_loss[6]))
            self.logging('-' * 75)

        return best_loss

    def save(self, path,epoch):
        self.logging("saving to %s" % path)
        model_path = os.path.join(path, f"model-ep{epoch}.pt")
        torch.save(self.vae.state_dict(), model_path)

    def load(self, path,epoch):
        model_path = os.path.join(path, f"model-ep{epoch}.pt")
        self.vae.load_state_dict(torch.load(model_path),strict=False)

