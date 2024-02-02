# Copyright (c) 2020-present, Royal Bank of Canada.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import torch
import random
import math
import json
import numpy as np
import os
from torch.utils.data import DataLoader, Dataset, Sampler, SequentialSampler, RandomSampler
from torch.utils.data.distributed import DistributedSampler
from torch.nn.utils.rnn import pad_sequence
from os import system
import logging
from collections import defaultdict, OrderedDict
from tqdm import tqdm

# root_path = "../data/"
# domain_dict = {"imdb":"0","yelp":"1","amazon":"2","yahoo":"3"}
# domain_i2d = {"0":"imdb","1": "yelp","2":"amazon","3":"yahoo"}
domain_i2d = {"0":"music","1":"family"}
domain_dict = {"music":0,"family":1}
root_path = "./data"

logger = logging.getLogger(__name__)

class BucketingMultipleFiles_DataLoader(object):
    def __init__(self, file_path, batch_size, max_seq_length, tokenizer, args, bucket=100, shuffle=True):

        self.batch_size = batch_size
        self.max_len = max_seq_length
        self.bucket_size = bucket * batch_size
        self.shuffle = shuffle
        self.file_path = file_path
        self.tokenizer = tokenizer
        self.args = args

        # prepare for the first file
        self.file_idx = 0
        self.cached_features_file = os.path.join(self.file_path, args.dataset.lower()+f'.segmented.nltk.split.seq64.{self.file_idx}.json' )
        self.dataset = PreVaeData(tokenizer, self.args, self.cached_features_file, block_size=self.args.block_size)
        self.num_examples = len(self.dataset.examples)
        self.num_batches = self.num_examples//batch_size
        self.example_lengths = [example['bert_token_length'] for example in self.dataset.examples]


    def __iter__(self):
        
        # sampler = BucketSampler(self.example_lengths, self.bucket_size, self.batch_size, droplast=True, shuffle=self.shuffle)
        # loader = DataLoader(self.dataset, batch_sampler=sampler, num_workers=0, collate_fn=PreparedTokenDataset.collate)

        # distributed
        sampler = DistributedSampler(self.dataset)
        loader = DataLoader(self.dataset, sampler=sampler, batch_size=self.batch_size, pin_memory=True, num_workers=0, collate_fn=PreVaeData.collate)
        yield from loader

        # update file name for next file
        self.file_idx += 1
        self.cached_features_file = os.path.join(self.file_path, self.args.dataset.lower()+f'.segmented.nltk.split.seq64.{self.file_idx}.json' )
        self.dataset = PreVaeData(self.tokenizer, self.args, self.cached_features_file, block_size=self.args.block_size)
        self.num_examples = len(self.dataset.examples)
        self.num_batches = self.num_examples//self.batch_size
        self.example_lengths = [example['bert_token_length'] for example in self.dataset.examples]


    def __len__(self):
        return self.num_batches

    def __del__(self):
        pass

    def reset(self):
        self.file_idx = 0
        
def build_dataload_and_cache_examples(args, tokenizer, evaluate=False):
    if isinstance(tokenizer, list):
        args.batch_size = args.bsz
        file_path=args.train_data_file
        dataloader = BucketingMultipleFiles_DataLoader(file_path, args.batch_size, args.max_seq_length, tokenizer, args, bucket=100, shuffle=True)
    else:
        pass 
    return dataloader

def pair_data_path(file_path):
    #write to file as the original input file form
    lines = open(file_path,"r").readlines()
    src_file_path = os.path.join(file_path[:-11],"src_file.txt")
    tar_file_path = os.path.join(file_path[:-11],"tar_file.txt")
    src_file = open(src_file_path,"w+")
    tar_file = open(tar_file_path,"w+")
    label = file_path[-1]
    domain = file_path.split("/")[4]
    for line in lines:
        sent_pair = line.split("\t")
        src_file.writelines(label+"\t"+sent_pair[0].strip()+"\t"+domain_dict[domain]+"\n")
        tar_file.writelines(str(1-int(label))+"\t"+sent_pair[1].strip()+"\t"+domain_dict[domain]+"\n")
    print(f"write src and tar file to \n{src_file_path} \n{tar_file_path}")
    return src_file_path, tar_file_path

def TokenFeat(sents, tokenizers, model,device):
    input_samples = []
    for line in sents:
        sent = line.split("\t")[1]
        tokenized_text0 = tokenizers.convert_tokens_to_ids(tokenizers.tokenize(sent))
        input_samples.append(tokenized_text0)
    tokenized_text0 = pad_sequence([torch.tensor(sample, dtype=torch.long) for sample in input_samples], batch_first=True, padding_value=bert_pad_token).to(device)
    bert_fea = model.vae.encoder(tokenized_text0)[1]
    mu, logvar = model.vae.encoder.linear(bert_fea).chunk(2, -1)
    return mu[:,100:]
        
def construct_domain_datasets(pretrain_domids,feat_type):
    #load multiple domain datasets
    ndomain = len(pretrain_domids)
    do_dataset = True
    if ndomain > 1:
        print("Training on %s domains"%ndomain)
        data_files_command = "cat"
        feat_all_splits = []
        # for split in ["train","dev","test"]:
        for split in ["train","dev","test"]:
            if os.path.exists(f"{root_path}/%s_all_data.txt"%split):
                print("%s dataset is built"%split)
                do_dataset = False
            data_files_command = "cat"
            feat_all = []
            for domid in pretrain_domids:
                data_name = domain_i2d[domid]
                data_pth = f"{root_path}/%s" %data_name
                try:
                    feat_pth = os.path.join(data_pth, "%s_%s.npy" %(split,feat_type))
                    data_files_command += " "+data_pth+"/%s_data.txt"%split
                    feat = np.load(feat_pth)#[N,300]
                except:
                    feat_pth = os.path.join(data_pth, "%s_%s.npy" %("test",feat_type))
                    data_files_command += " "+data_pth+"/%s_data.txt"%split
                    feat = np.load(feat_pth)#[N,300]
                feat_all.append(feat)
            feat_all_splits.append(np.concatenate(([feat for feat in feat_all])))
            # concate the different pretrained datasets
            if do_dataset:
                data_files_command += f" > {root_path}/%s_all_data.txt"%split
                system(data_files_command)
            train_data_pth = os.path.join(root_path, "train_all_data.txt")
            dev_data_pth = os.path.join(root_path, "dev_all_data.txt")
            test_data_pth = os.path.join(root_path, "test_all_data.txt")
        train_feat = feat_all_splits[0]
        dev_feat = feat_all_splits[1]
        test_feat = feat_all_splits[2]
    else:
        data_name = domain_i2d[pretrain_domids[0]]
        train_data_pth = os.path.join(root_path+"%s/"%data_name, "train_data.txt")
        train_feat_pth = os.path.join(root_path+"%s/"%data_name, "train_%s.npy" % feat)
        train_feat = np.load(train_feat_pth)
        dev_data_pth = os.path.join(root_path+"%s/"%data_name, "dev_data.txt")
        dev_feat_pth = os.path.join(root_path+"%s/"%data_name, "dev_%s.npy" %feat)
        dev_feat = np.load(dev_feat_pth)
        test_data_pth = os.path.join(root_path+"%s/"%data_name, "test_data.txt")
        test_feat_pth = os.path.join(root_path+"%s/"%data_name, "test_%s.npy" %feat)
        test_feat = np.load(test_feat_pth)
    
    return train_data_pth, dev_data_pth, test_data_pth, train_feat, dev_feat, test_feat


class VocabEntry(object):
    def __init__(self, vocab_size=100000):
        super(VocabEntry, self).__init__()
        self.vocab_size = vocab_size

        self.word2id = OrderedDict()
        self.unk_id = 3
        self.word2id['<pad>'] = 0
        self.word2id['<s>'] = 1
        self.word2id['</s>'] = 2
        self.word2id['<unk>'] = self.unk_id
        self.id2word_ = list(self.word2id.keys())

    def create_glove_embed(self, glove_file="../glove.840B.300d.txt"):
        self.glove_embed = np.random.randn(len(self) - 4, 300)
        with open(glove_file) as f:
            for line in f:
                word, vec = line.split(' ', 1)

                wid = self[word]
                if wid > self.unk_id:
                    v = np.fromstring(vec, sep=" ", dtype=np.float32)
                    self.glove_embed[wid - 4, :] = v

        _mu = self.glove_embed.mean()
        _std = self.glove_embed.std()
        self.glove_embed = np.vstack([np.random.randn(4, self.glove_embed.shape[1]) * _std + _mu,
                                      self.glove_embed])

    def __getitem__(self, word):
        idx = self.word2id.get(word, self.unk_id)
        return idx if idx < self.vocab_size else self.unk_id

    def __contains__(self, word):
        return word in self.word2id

    def __len__(self):
        return min(len(self.word2id), self.vocab_size)

    def id2word(self, wid):
        return self.id2word_[wid]

    def decode_sentence(self, sentence):
        decoded_sentence = []
        for wid_t in sentence:
            wid = wid_t.item()
            decoded_sentence.append(self.id2word_[wid])
        return decoded_sentence

    def build(self, sents,vocab_size=9999):
        wordcount = defaultdict(int)
        for sent in sents:
            for w in sent:
                wordcount[w] += 1
        sorted_words = sorted(wordcount, key=wordcount.get, reverse=True)

        for idx, word in enumerate(sorted_words[:vocab_size]):
            self.word2id[word] = idx + 4
        self.id2word_ = list(self.word2id.keys())

class BucketSampler(Sampler):
    def __init__(self, lens, bucket_size, batch_size, droplast=False, shuffle=True):
        self._lens = lens
        self._batch_size = batch_size
        self._bucket_size = bucket_size
        self._droplast = droplast
        self._shuf = shuffle

    def __iter__(self):
        ids = list(range(len(self._lens)))
        if self._shuf:
            random.shuffle(ids)
        # buckets = [sorted(ids[i:i+self._bucket_size],
        #                   key=lambda i: self._lens[i], reverse=True)
        #            for i in range(0, len(ids), self._bucket_size)]
        buckets = [ids[i:i+self._bucket_size] for i in range(0, len(ids), self._bucket_size)]          
        batches = [bucket[i:i+self._batch_size]
                   for bucket in buckets
                   for i in range(0, len(bucket), self._batch_size)]
        if self._droplast:
            batches = [batch for batch in batches
                       if len(batch) == self._batch_size]
        if self._shuf:
            random.shuffle(batches)
        return iter(batches)

    def __len__(self):
        bucket_sizes = ([self._bucket_size]
                        * (len(self._lens) // self._bucket_size)
                        + [len(self._lens) % self._bucket_size])
        if self._droplast:
            return sum(s//self._batch_size for s in bucket_sizes)
        else:
            return sum(math.ceil(s/self._batch_size) for s in bucket_sizes)
        
class BucketingDataLoader(object):
    def __init__(self, file_path, batch_size, max_seq_length, encoder_tokenizer,decoder_tokenizer, args, bucket=100, shuffle=False):

        self.dataset = PreVaeData(encoder_tokenizer,decoder_tokenizer, args, file_path, block_size=-1,max_length=max_seq_length)
        self.batch_size = batch_size
        self.max_len = max_seq_length
        self.bucket_size = bucket * batch_size
        self.shuffle = shuffle
        self.num_examples = len(self.dataset.examples)
        self.num_batches = self.num_examples//batch_size
        self.example_lengths = [example['bert_token_length'] for example in self.dataset.examples]

    def __iter__(self):
        sampler = BucketSampler(self.example_lengths, self.bucket_size, self.batch_size, droplast=True, shuffle=self.shuffle)
        loader = DataLoader(self.dataset, batch_sampler=sampler, num_workers=0, collate_fn=PreVaeData.collate)
        yield from loader

    def __len__(self):
        return self.num_batches

    def __del__(self):
        pass

class PreVaeData(object):
    def __init__(self,encoder_tokenizer, decoder_tokenizer,args,fname,block_size=-1,max_length=100):

        assert os.path.isfile(fname)
        directory, filename = os.path.split(fname)
        cached_features_file = os.path.join(directory, f'cached_lm_gpt_bert_{block_size}_{filename[:-4]}.json')
        # Bert tokenizer special tokens
        self.bert_pad_token=encoder_tokenizer.convert_tokens_to_ids([encoder_tokenizer.pad_token])[0]

        # GPT-2 tokenizer special tokens
        self.gpt2_pad_token=decoder_tokenizer.convert_tokens_to_ids([decoder_tokenizer.pad_token])[0]
        self.gpt2_bos_token=decoder_tokenizer.convert_tokens_to_ids([decoder_tokenizer.bos_token])[0]
        self.gpt2_eos_token=decoder_tokenizer.convert_tokens_to_ids([decoder_tokenizer.eos_token])[0]
        
        global bert_pad_token
        global gpt2_pad_token
        
        gpt2_pad_token = self.gpt2_pad_token
        bert_pad_token = self.bert_pad_token
        
        if os.path.exists(cached_features_file):
            logger.info("Loading features from cached file %s", cached_features_file)
            with open(cached_features_file, 'r') as handle:
                self.examples = json.load(handle)
        else:
            dropped, count =self._read_corpus_pretainvae(fname,encoder_tokenizer,decoder_tokenizer,max_length=max_length)
            print("The number of dropped sentences is %d", dropped)
            print("The number of processed sentences is %d", count)
            
            logger.info("Saving features into cached file %s", cached_features_file)
            with open(cached_features_file, 'w') as handle:
                json.dump(self.examples, handle)
                
    
    def __len__(self):
        return len(self.examples)
    def __getitem__(self, item):
        return self.examples[item]

    @staticmethod
    def collate(examples):
        # Convert to Tensors and build dataset
        input_ids_bert = pad_sequence([torch.tensor(f['bert_token'], dtype=torch.long) for f in examples], batch_first=True, padding_value=bert_pad_token)
        input_ids_gpt = pad_sequence([torch.tensor(f['gpt2_token'], dtype=torch.long) for f in examples], batch_first=True, padding_value=gpt2_pad_token)
        token_lengths = torch.tensor( [[f['bert_token_length'], f['gpt2_token_length']] for f in examples] , dtype=torch.long)
        input_domids = torch.tensor([torch.tensor(f['domid'],dtype=torch.long) for f in examples])

        return (input_ids_bert, input_ids_gpt, token_lengths,input_domids)

    def _read_corpus_pretainvae(self,fname,encoder_tokenizer,decoder_tokenizer,max_length):
        with open(fname) as fin:
            self.examples = []
            dropped,count=0,0
            for i,line in enumerate(tqdm(fin,desc='Datasets building')): 
                split_line = line.strip().split('\t')
                domid = split_line[-1]
                lb = split_line[0]
                split_line_text = split_line[1]
                if len(split_line_text.split()) < 1:
                    dropped += 1
                    continue

                if max_length:
                    if len(split_line_text.split()) > max_length:
                        dropped += 1
                        continue
                
                tokenized_text0 = encoder_tokenizer.convert_tokens_to_ids(encoder_tokenizer.tokenize(split_line_text))
                # tokenized_text0 = encoder_tokenizer.add_special_tokens_single_sentence(tokenized_text0)
                tokenized_text0_length = len(tokenized_text0) 

                tokenized_text1 = decoder_tokenizer.convert_tokens_to_ids(decoder_tokenizer.tokenize(split_line_text))
                # tokenized_text1 = decoder_tokenizer.add_special_tokens_single_sentence(tokenized_text1)
                tokenized_text1 = [self.gpt2_bos_token] + tokenized_text1 + [self.gpt2_eos_token]
                tokenized_text1_length = len(tokenized_text1)
                
                example = {
                        'bert_token': tokenized_text0,
                        'bert_token_length':tokenized_text0_length,
                        'gpt2_token':tokenized_text1,
                        'gpt2_token_length': tokenized_text1_length,
                        'domid':int(domid),
                    }
                self.examples.append(example)
                count +=1
        return dropped, count
    
            
class MonoTextData(object):
    def __init__(self, fname, n_domains=1, max_length=None, vocab=None, glove=False,vocab_size=10000):
        super(MonoTextData, self).__init__()
        self.data, self.vocab, self.dropped, self.labels, self.domids = self._read_corpus(
            fname, n_domains, max_length, vocab, glove,vocab_size=9999)

    def __len__(self):
        return len(self.data)

    def _read_corpus(self, fname, n_domains, max_length, vocab, glove,vocab_size=9999):
        data = []
        labels = []
        domids = []
        dropped = 0

        sents = []
        with open(fname) as fin:
            domid = False
            for line in fin:
                if n_domains >1:
                    split_line = line.strip().split('\t')
                    # if len(split_line)>2: #domain idx is included
                    domid = split_line[-1]
                    lb = split_line[0]
                    split_line = " ".join(split_line[1:-1]).split() #split into tokens
                else:
                    split_line = line.strip().split('\t')
                    lb = split_line[0]
                    if len(split_line)<2:
                        split_line=["i", "do", "not","know","the","result"]
                    else:
                        split_line = split_line[1].split()
                    domid = 0

                if len(split_line) < 1:
                    dropped += 1
                    continue

                if max_length:
                    if len(split_line) > max_length:
                        dropped += 1
                        continue

                labels.append(int(lb))
                domids.append(int(domid))
                sents.append(split_line)
                data.append(split_line)

        if isinstance(vocab, int):
            vocab = VocabEntry(vocab)
            vocab.build(sents,vocab_size=vocab_size)
            if glove:
                vocab.create_glove_embed()
        elif vocab is None:
            vocab = VocabEntry()
            vocab.build(sents,vocab_size=vocab_size)
            if glove:
                vocab.create_glove_embed()

        data = [[vocab[word] for word in x] for x in data] #

        return data, vocab, dropped, labels, domids


    def _to_tensor(self, batch_data, batch_first, device, min_len=0):
        batch_data = [sent + [self.vocab['</s>']] for sent in batch_data]
        sents_len = [len(sent) for sent in batch_data]
        max_len = max(sents_len)
        max_len = max(min_len, max_len)
        batch_size = len(sents_len)
        sents_new = []
        sents_new.append([self.vocab['<s>']] * batch_size)
        for i in range(max_len):
            sents_new.append([sent[i] if len(sent) > i else self.vocab['<pad>']
                              for sent in batch_data])
        sents_ts = torch.tensor(sents_new, dtype=torch.long,
                                requires_grad=False, device=device)

        if batch_first:
            sents_ts = sents_ts.permute(1, 0).contiguous()

        return sents_ts, [length + 1 for length in sents_len]

    def data_iter(self, batch_size, device, batch_first=False, shuffle=True):
        index_arr = np.arange(len(self.data))
        if shuffle:
            np.random.shuffle(index_arr)
        batch_num = int(np.ceil(len(index_arr)) / float(batch_size))
        for i in range(batch_num):
            batch_ids = index_arr[i * batch_size: (i + 1) * batch_size]
            batch_data = [self.data[index] for index in batch_ids]
            batch_data, sents_len = self._to_tensor(batch_data, batch_first, device)
            yield batch_data, sents_len

    def create_data_batch_labels(self, batch_size, device, batch_first=False, min_len=5):
        sents_len = np.array([len(sent) for sent in self.data])
        sort_idx = np.argsort(sents_len)
        sort_len = sents_len[sort_idx]

        change_loc = []
        for i in range(1, len(sort_len)):
            if sort_len[i] != sort_len[i - 1]:
                change_loc.append(i)
        change_loc.append(len(sort_len))

        batch_data_list = []
        batch_label_list = []
        total = 0
        cur = 0
        for idx in change_loc:
            while cur < idx:
                batch_data = []
                batch_label = []
                nxt = min(cur + batch_size, idx)
                for id_ in range(cur, nxt):
                    batch_data.append(self.data[sort_idx[id_]])
                    batch_label.append(self.labels[sort_idx[id_]])
                cur = nxt
                batch_data, sents_len = self._to_tensor(batch_data, batch_first, device, min_len)
                batch_data_list.append(batch_data)
                batch_label_list.append(batch_label)

                total += batch_data.size(0)
                assert sents_len == ([sents_len[0]] * len(sents_len))

        return batch_data_list, batch_label_list
    
    def create_data_batch_domids(self, batch_size, device, batch_first=False, min_len=5):
        sents_len = np.array([len(sent) for sent in self.data])
        sort_idx = np.argsort(sents_len)
        sort_len = sents_len[sort_idx]

        change_loc = []
        for i in range(1, len(sort_len)):
            if sort_len[i] != sort_len[i - 1]:
                change_loc.append(i)
        change_loc.append(len(sort_len))

        batch_data_list = []
        batch_domid_list = []
        batch_label_list = []
        total = 0
        cur = 0
        for idx in change_loc:
            while cur < idx:
                batch_data = []
                batch_domid = []
                batch_label = []
                nxt = min(cur + batch_size, idx)
                for id_ in range(cur, nxt):
                    batch_data.append(self.data[sort_idx[id_]])
                    batch_domid.append(self.domids[sort_idx[id_]])
                    batch_label.append(self.labels[sort_idx[id_]])
                cur = nxt
                batch_data, sents_len = self._to_tensor(batch_data, batch_first, device, min_len)
                batch_data_list.append(batch_data)
                batch_domid_list.append(batch_domid)
                batch_label_list.append(batch_label)

                total += batch_data.size(0)
                assert sents_len == ([sents_len[0]] * len(sents_len))

        return batch_data_list, batch_domid_list, batch_label_list

    def create_data_batch_feats(self, batch_size, feats, device, batch_first=False,min_len=5):
        sents_len = np.array([len(sent) for sent in self.data])
        print("Maximum length: %d" % max(sents_len))
        sort_idx = np.argsort(sents_len)
        sort_len = sents_len[sort_idx]

        change_loc = []
        for i in range(1, len(sort_len)):
            if sort_len[i] != sort_len[i - 1]:
                change_loc.append(i)
        change_loc.append(len(sort_len))

        batch_data_list = []
        batch_feat_list = []
        batch_domid_list = []
        batch_label_list = []
        total = 0
        cur = 0
        for idx in change_loc:
            while cur < idx:
                batch_encoder_data,batch_decoder_data = [], []
                batch_feat = []
                batch_domid = []
                batch_label = []
                nxt = min(cur + batch_size, idx)
                for id_ in range(cur, nxt):
                    batch_encoder_data.append(self.data[sort_idx[id_]])
                    # batch_decoder_data.append(self.data[sort_idx[id_]])
                    batch_feat.append(feats[sort_idx[id_]])
                    batch_domid.append(self.domids[sort_idx[id_]])
                    batch_label.append(self.labels[sort_idx[id_]])
                cur = nxt
                batch_encoder_data, sents_len = self._to_tensor(batch_encoder_data, batch_first, device,min_len)
                batch_data_list.append(batch_encoder_data)
                batch_feat = torch.tensor(
                    np.array(batch_feat), dtype=torch.float, requires_grad=False, device=device)
                batch_feat_list.append(batch_feat)
                batch_domid = torch.tensor(
                    np.array(batch_domid), dtype=torch.long, requires_grad=False, device=device)
                batch_domid_list.append(batch_domid)
                # batch_label = torch.tensor(
                    # np.array(batch_label), dtype=torch.long, requires_grad=False, device=device)
                batch_label_list.append(batch_label)

                total += batch_encoder_data.size(0)
                assert sents_len == ([sents_len[0]] * len(sents_len))

        return batch_data_list, batch_feat_list, batch_domid_list, batch_label_list
    
    def create_data_batch(self, batch_size, feats, device, batch_first=False):
        sents_len = np.array([len(sent) for sent in self.data])
        print("Maximum length: %d" % max(sents_len))
        sort_idx = np.argsort(sents_len)
        sort_len = sents_len[sort_idx]

        change_loc = []
        for i in range(1, len(sort_len)):
            if sort_len[i] != sort_len[i - 1]:
                change_loc.append(i)
        change_loc.append(len(sort_len))

        batch_data_list = []
        batch_feat_list = []
        batch_domid_list = []
        batch_label_list = []
        total = 0
        cur = 0
        for idx in change_loc:
            while cur < idx:
                batch_data = []
                batch_feat = []
                batch_domid = []
                batch_label = []
                nxt = min(cur + batch_size, idx)
                for id_ in range(cur, nxt):
                    batch_data.append(self.data[sort_idx[id_]])
                    batch_feat.append(feats[sort_idx[id_]])
                    batch_domid.append(self.domids[sort_idx[id_]])
                    batch_label.append(self.labels[sort_idx[id_]])

                cur = nxt
                batch_data, sents_len = self._to_tensor(batch_data, batch_first, device)
                batch_data_list.append(batch_data)
                batch_feat = torch.tensor(
                    np.array(batch_feat), dtype=torch.float, requires_grad=False, device=device)
                batch_feat_list.append(batch_feat)
                batch_domid = torch.tensor(
                    np.array(batch_domid), dtype=torch.long, requires_grad=False, device=device)
                batch_domid_list.append(batch_domid)
                batch_label = torch.tensor(
                    np.array(batch_label), dtype=torch.long, requires_grad=False, device=device)
                batch_label_list.append(batch_label)

                total += batch_data.size(0)
                assert sents_len == ([sents_len[0]] * len(sents_len))

        return batch_data_list, batch_feat_list, batch_domid_list,batch_label_list

    def data_sample(self, nsample, device, batch_first=False, shuffle=True):
        index_arr = np.arange(len(self.data))
        if shuffle:
            np.random.shuffle(index_arr)
        batch_ids = index_arr[:nsample]
        batch_data = [self.data[index] for index in batch_ids]

        batch_data, sents_len = self._to_tensor(batch_data, batch_first, device)

        return batch_data, sents_len

    def create_data_batch_pretrainvae(self,tokenizers,batch_size,device,min_len=5):
        # Bert tokenizer special token
        

        
        input_ids_bert = pad_sequence([torch.tensor(f, dtype=torch.long) for f in self.data[0]], batch_first=True, padding_value=bert_pad_token)
        input_ids_gpt = pad_sequence([torch.tensor(f, dtype=torch.long) for f in self.data[1]], batch_first=True, padding_value=gpt2_pad_token)
        token_lengths = torch.tensor( [[len(f[0]), len(f[1])] for f in self.data] , dtype=torch.long)
        
        return input_ids_bert,input_ids_gpt,token_lengths