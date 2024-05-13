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
from .influential_vae_s import DecomposedVAE as HiVAE
import numpy as np
import time
import os

        # "train": train,
        # "valid": dev,
        # "test": test,
        # "feat": feat,
        # "bsz": args.bsz,
        # "save_path": save_path,
        # "logging": logging,
        # "text_only": args.text_only,
        # "n_domains":args.wdomid,
        # "train_schema":args.train_schema,
        # "wdomid":args.wdomid,
        # "sSparsity":args.sSparsity,
        # "cSparsity":args.cSparsity,
        # "lambda_mask":args.lambda_mask,

class DecomposedVAE:
    def __init__(self, train, valid, test, feat, bsz, save_path, logging, log_interval, num_epochs,
                 enc_lr, dec_lr, warm_up, kl_start, beta1, beta2, srec_weight, reg_weight,
                 aggressive, text_only,n_domains,train_schema,wdomid,sSparsity,cSparsity,lambda_mask,vae_params):
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
        
        self.train_data, self.train_feat, self.train_domid,_  = train
        self.valid_data, self.valid_feat, self.valid_domid,_ = valid
        self.test_data, self.test_feat,self.test_domid,_ = test
        self.feat = feat
        if vae_params["flow_type"] == "True":
            self.vae = HiVAE(**vae_params)
        else:
            self.vae = VAE(**vae_params) 
        if self.use_cuda:
            self.vae.cuda()

        self.enc_params = list(self.vae.lstm_encoder.parameters()) + \
            list(self.vae.mlp_encoder.parameters())
        self.enc_optimizer = optim.Adam(self.enc_params, lr=self.enc_lr)
        self.dec_optimizer = optim.SGD(self.vae.decoder.parameters(), lr=self.dec_lr)

        self.nbatch = len(self.train_data)
        self.anneal_rate = (1.0 - kl_start) / (warm_up * self.nbatch)

    def train(self, epoch):
        self.vae.train()

        total_rec_loss = 0
        total_kl1_loss = 0
        total_kl2_loss = 0
        total_srec_loss = 0
        start_time = time.time()
        step = 0
        num_words = 0
        num_sents = 0

        for idx in np.random.permutation(range(self.nbatch)):
            batch_data = self.train_data[idx]
            batch_feat = self.train_feat[idx]
            sent_len, batch_size = batch_data.size()
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

            # sub_iter = 1
            # batch_data_enc = batch_data
            # batch_feat_enc = batch_feat
            # burn_num_words = 0
            # burn_pre_loss = 1e4
            # burn_cur_loss = 0
            # while self.aggressive and sub_iter < 100:
            #     self.enc_optimizer.zero_grad()
            #     self.dec_optimizer.zero_grad()

            #     target_enc = batch_data_enc[1:]
            #     burn_sent_len, burn_batch_size = batch_data_enc.size()
            #     burn_num_words += (burn_sent_len - 1) * burn_batch_size

            #     logits, kl1_loss, kl2_loss, reg_ic, style_logit = self.vae.loss(batch_data_enc, batch_feat_enc,u=batch_domid)
            #     logits = logits.view(-1, logits.size(2))
            #     rec_loss = F.cross_entropy(logits, target_enc.view(-1), reduction="none")
            #     rec_loss = rec_loss.view(-1, burn_batch_size).sum(0)
            #     loss = rec_loss + beta1 * kl1_loss + beta2 * kl2_loss
                
            #     loss = loss + 0.0001*reg_ic[0]

            #     burn_cur_loss = loss.sum().item()
            #     loss = loss.mean(dim=-1)

            #     loss.backward(retain_graph=True)
            #     torch.nn.utils.clip_grad_norm_(self.enc_params, 0.1)
            #     torch.nn.utils.clip_grad_norm_(self.vae.decoder.parameters(), 5.0)

            #     self.enc_optimizer.step()

            #     id_ = np.random.randint(self.nbatch)
            #     batch_data_enc = self.train_data[id_]
            #     batch_feat_enc = self.train_feat[id_]
            #     burn_sent_len, burn_batch_size = batch_data_enc.size()
            #     shift = np.random.randint(max(1, burn_sent_len - 9))
            #     batch_data_enc = batch_data_enc[shift:min(burn_sent_len, shift + 10), :]

            #     if sub_iter % 15 == 0:
            #         burn_cur_loss = burn_cur_loss / burn_num_words
            #         if burn_pre_loss - burn_cur_loss < 0:
            #             break
            #         burn_pre_loss = burn_cur_loss
            #         burn_cur_loss = burn_num_words = 0

            #     sub_iter += 1

            self.enc_optimizer.zero_grad()
            self.dec_optimizer.zero_grad()

            vae_logits, vae_kl1_loss, vae_kl2_loss, sparse_loss,cmask_loss = self.vae.loss(
                x=batch_data, feat=batch_feat,u=batch_domid, no_ic=self.sSparsity == 0)
            vae_logits = vae_logits.view(-1, vae_logits.size(2))
            vae_rec_loss = F.cross_entropy(vae_logits, target.view(-1), reduction="none")
            vae_rec_loss = vae_rec_loss.view(-1, batch_size).sum(0)

            vae_loss = vae_rec_loss + beta1 * vae_kl1_loss + beta2 * vae_kl2_loss
                
            vae_loss = vae_loss.mean()
            total_rec_loss += vae_rec_loss.sum().item()
            total_kl1_loss += vae_kl1_loss.sum().item()
            total_kl2_loss += vae_kl2_loss.sum().item()
            loss = loss + vae_loss
            try:
                loss = loss + 0.0001*sparse_loss[0]+0.0001*sparse_loss[1]+0.0001*cmask_loss
            except:
                loss = loss + 0.0001*sparse_loss[0]+0.0001*cmask_loss
            idx = np.random.choice(self.feat.shape[1], batch_size * 10)
            neg_feat = torch.tensor(self.feat[idx], dtype=torch.float,
                                        requires_grad=False, device=self.device)
            srec_loss, reg_loss, srec_raw_loss = self.vae.var_loss(batch_feat, neg_feat, 10)
            total_srec_loss += srec_raw_loss.item()
            if self.train_schema == "cpvae":
                loss = loss + self.srec_weight * srec_loss + self.reg_weight * reg_loss

            loss.backward()

            nn.utils.clip_grad_norm_(self.vae.parameters(), 5.0)
            if not self.aggressive:
                self.enc_optimizer.step()
            self.dec_optimizer.step()

            if step % self.log_interval == 0 and step > 0:
                cur_rec_loss = total_rec_loss / num_sents
                cur_kl1_loss = total_kl1_loss / num_sents
                cur_kl2_loss = total_kl2_loss / num_sents
                cur_vae_loss = cur_rec_loss + cur_kl1_loss + cur_kl2_loss
                cur_srec_loss = total_srec_loss / num_sents
                # cur_cls_loss = total_cls_loss / num_sents
                # print(acc_list)
                # cur_acc = sum(acc_list)/len(acc_list)
                # cur_acc = 0
                elapsed = time.time() - start_time
                self.logging(
                    '| epoch {:2d} | {:5d}/{:5d} batches | {:5.2f} ms/batch | loss {:3.2f} | '
                    'recon {:3.2f} | kl1 {:3.2f} | kl2 {:3.2f}'.format(
                        epoch, step, self.nbatch, elapsed * 1000 / self.log_interval, cur_vae_loss,
                        cur_rec_loss, cur_kl1_loss, cur_kl2_loss))
                total_rec_loss = 0
                total_kl1_loss = 0
                total_kl2_loss = 0
                total_srec_loss = 0
                total_cls_loss = 0
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

                vae_logits, vae_kl1_loss, vae_kl2_loss, sparse_loss, cmask_loss = self.vae.loss(
                    batch_data, batch_feat,u=batch_domid)
                vae_logits = vae_logits.view(-1, vae_logits.size(2))
                vae_rec_loss = F.cross_entropy(vae_logits, target.view(-1), reduction="none")
                # vae_cls_loss = F.cross_entropy(style_logits,batch_label)
                total_rec_loss += vae_rec_loss.sum().item()
                total_kl1_loss += vae_kl1_loss.sum().item()
                total_kl2_loss += vae_kl2_loss.sum().item()
                # total_cls_loss += vae_cls_loss.sum().item()

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
                self.save(self.save_path)
                best_loss = vae_loss

            if vae_loss > self.opt_dict["best_loss"]:
                self.opt_dict["not_improved"] += 1
                if self.opt_dict["not_improved"] >= 2 and epoch >= 15:
                    self.opt_dict["not_improved"] = 0
                    self.opt_dict["lr"] = self.opt_dict["lr"] * 0.5
                    self.load(self.save_path)
                    decay_cnt += 1
                    self.dec_optimizer = optim.SGD(
                        self.vae.decoder.parameters(), lr=self.opt_dict["lr"])
            else:
                self.opt_dict["not_improved"] = 0
                self.opt_dict["best_loss"] = vae_loss

            if decay_cnt == 5:
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

    def save(self, path):
        self.logging("saving to %s" % path)
        model_path = os.path.join(path, "model.pt")
        torch.save(self.vae.state_dict(), model_path)

    def load(self, path):
        model_path = os.path.join(path, "model.pt")
        self.vae.load_state_dict(torch.load(model_path),strict=False)
