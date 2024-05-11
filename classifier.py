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
import argparse
from utils.text_utils_our import MonoTextData
import numpy as np
import os
from os import system

class CNNClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, filter_sizes, n_filters, dropout):
        super(CNNClassifier, self).__init__()
        self.n_filters = n_filters

        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.cnns = nn.ModuleList([
            nn.Conv2d(embed_dim, n_filters, (x, 1)) for x in filter_sizes])
        self.dropout = nn.Dropout(dropout)
        self.output = nn.Linear(len(filter_sizes) * n_filters, 1)
        # self.encoder_transform = nn.Linear(16,len(filter_sizes) * n_filters)#16 

    def forward(self, inputs):
        inputs = self.embedding(inputs).unsqueeze(-1)
        inputs = inputs.permute(0, 2, 1, 3)
        outputs = []
        for cnn in self.cnns:
            conv = cnn(inputs)
            h = F.leaky_relu(conv)
            pooled = torch.max(h, 2)[0].view(-1, self.n_filters)
            outputs.append(pooled)
        outputs = torch.cat(outputs, -1)
        outputs = self.dropout(outputs)
        logits = self.output(outputs)
        return logits.squeeze(1)        

def evaluate_func(model, eval_data, eval_label):
    correct_num = 0
    total_sample = 0
    acc_list = []
    for batch_data, batch_label in zip(eval_data, eval_label):
        batch_size = batch_data.size(0)
        logits = model(batch_data)
        probs = torch.sigmoid(logits)
        # y_hat = torch.argmax(logits,dim=-1)
        y_hat = list((probs > 0.5).long().cpu().numpy())
        correct_num += sum([p == q for p, q in zip(batch_label, y_hat)])
        total_sample += batch_size
        prob = probs.detach().cpu().numpy()
        acc_list.extend([p if y>0  else 1-p for y, p in zip(batch_label,prob)])
    return correct_num / total_sample, acc_list

def main(args):
    root_path = "/mnt/Data3/hanqiyan/UDA/real_world/data/"
    domain_dict = {"imdb":0,"yelp_dast":1,"amazon":2,"yahoo":3}
    domain_i2d = {"0":"imdb","1": "yelp_dast","2":"amazon","3":"yahoo"}
    pretrain_domids = ["0","1","2","3"]
    args.n_domains = len(pretrain_domids)

    #load multiple domain datasets
    do_dataset = True
    if args.n_domains > 1:
        data_files_command = "cat"
        feat_all_splits = []
        for split in ["train","dev","test"]:
            do_dataset = True
            if os.path.exists(root_path+"/%s_all_data.txt"%split):
                print("%s dataset is built"%split)
                do_dataset = False
            data_files_command = "cat"
            feat_all = []
            for domid in pretrain_domids:
                data_name = domain_i2d[domid]
                data_pth = root_path+"/%s" %data_name
                # feat_pth = os.path.join(data_pth, "%s_%s.npy" %(split,args.feat))
                data_files_command += " "+data_pth+"/%s_data.txt"%split
                # feat = np.load(feat_pth)#[N,300]
                # feat_all.append(feat)
            # feat_all_splits.append(np.concatenate(([feat for feat in feat_all])))
            # concate the different pretrained datasets
            if do_dataset:
                print("Constructing %s %s"%(data_name,split))
                data_files_command += " > %s/%s_all_data.txt"%(root_path,split)
                system(data_files_command)
            train_data_pth = os.path.join(root_path, "train_all_data.txt")
            dev_data_pth = os.path.join(root_path, "dev_all_data.txt")
            test_data_pth = os.path.join(root_path, "test_all_data.txt")
        # train_feat = feat_all_splits[0]
        # dev_feat = feat_all_splits[1]
        # test_feat = feat_all_splits[2]
    else:
        args.data_name = domain_i2d[pretrain_domids[0]]
        train_data_pth = os.path.join(root_path+"%s/"%args.data_name, "train_data.txt")
        # train_feat_pth = os.path.join(root_path+"%s/"%args.data_name, "train_%s.npy" % args.feat)
        # train_feat = np.load(train_feat_pth)
        dev_data_pth = os.path.join(root_path+"%s/"%args.data_name, "dev_data.txt")
        # dev_feat_pth = os.path.join(root_path+"%s/"%args.data_name, "dev_%s.npy" % args.feat)
        # dev_feat = np.load(dev_feat_pth)
        test_data_pth = os.path.join(root_path+"%s/"%args.data_name, "test_data.txt")
        # test_feat_pth = os.path.join(root_path+"%s/"%args.data_name, "test_%s.npy" % args.feat)
        # test_feat = np.load(test_feat_pth)

    train_data = MonoTextData(train_data_pth,args.n_domains)
    # assert len(train_data) == train_feat.shape[0]

    vocab = train_data.vocab
    print('Vocabulary size: %d' % len(vocab))

    dev_data = MonoTextData(dev_data_pth,args.n_domains, vocab=vocab)
    # assert len(dev_data) == dev_feat.shape[0]
    
    test_data = MonoTextData(test_data_pth, args.n_domains, vocab=vocab)  
    # assert len(test_data) == test_feat.shape[0]

    
    path = "checkpoint/%s-classifier-pure.pt" % args.data_name

    glove_embed = np.zeros((len(vocab), 300))
    with open("/mnt/Data3/hanqiyan/glove.840B.300d.txt") as f:
        for line in f:
            word, vec = line.split(' ', 1)
            if word in vocab:
                wid = vocab[word]
                glove_embed[wid, :] = np.fromstring(vec, sep=' ', dtype=np.float32)

        _mu = glove_embed.mean()
        _std = glove_embed.std()
        glove_embed[:4, :] = np.random.randn(4, 300) * _std + _mu

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_batch,_, train_label = train_data.create_data_batch_domids(128, device, batch_first=True)
    dev_batch,_, dev_label = dev_data.create_data_batch_domids(128, device, batch_first=True)
    test_batch,_, test_label = test_data.create_data_batch_domids(128, device, batch_first=True)

    model = CNNClassifier(len(vocab), 300, [1, 2, 3, 4, 5], 500, 0.5).to(device)
    optimizer = optim.Adam(model.parameters(), lr=5e-4)
    nbatch = len(train_batch)
    best_acc = 0.0
    step = 0

    with torch.no_grad():
        model.embedding.weight.fill_(0.)
        model.embedding.weight += torch.FloatTensor(glove_embed).to(device)

    for epoch in range(args.max_epochs):
        for idx in np.random.permutation(range(nbatch)):
            batch_data = train_batch[idx]
            batch_label = train_label[idx]
            batch_label = torch.tensor(batch_label, dtype=torch.float,
                                       requires_grad=False, device=device)

            optimizer.zero_grad()
            logits = model(batch_data)
            loss = F.binary_cross_entropy_with_logits(logits, batch_label)
            loss.backward()
            optimizer.step()

            step += 1
            if step % 1000 == 0:
                print('Loss: %2f' % loss.item())
                acc = evaluate_func(model, dev_batch, dev_label)
                print(acc)
        model.eval()
        acc = evaluate_func(model, dev_batch, dev_label)
        model.train()
        print('Valid Acc: %.2f' % acc)
        if acc > best_acc:
            best_acc = acc
            print('saving to %s' % path)
            torch.save(model.state_dict(), path)

    model.load_state_dict(torch.load(path))
    model.eval()
    acc = evaluate_func(model, test_batch, test_label)
    print('Test Acc: %.2f' % acc)

def add_args(parser):
    parser.add_argument('--data_name', type=str, default='yelp')
    parser.add_argument('--max_epochs', type=int, default=50)
    parser.add_argument('--n_domains', default=1, type=int,
                        help='if use multi-domain dataset')   


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_args(parser)
    args = parser.parse_args()

    main(args)
