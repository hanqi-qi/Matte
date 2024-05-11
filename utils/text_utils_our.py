# Copyright (c) 2020-present, Royal Bank of Canada.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import torch
import numpy as np

from collections import defaultdict, OrderedDict

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

    def create_glove_embed(self, glove_file="/mnt/Data3/hanqiyan/glove.840B.300d.txt"):
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

    def create_data_batch(self, batch_size, device, batch_first=False):
        sents_len = np.array([len(sent) for sent in self.data])
        sort_idx = np.argsort(sents_len)
        sort_len = sents_len[sort_idx]

        change_loc = []
        for i in range(1, len(sort_len)):
            if sort_len[i] != sort_len[i - 1]:
                change_loc.append(i)
        change_loc.append(len(sort_len))

        batch_data_list = []
        total = 0
        cur = 0
        for idx in change_loc:
            while cur < idx:
                batch_data = []
                nxt = min(cur + batch_size, idx)
                for id_ in range(cur, nxt):
                    batch_data.append(self.data[sort_idx[id_]])
                cur = nxt
                batch_data, sents_len = self._to_tensor(batch_data, batch_first, device)
                batch_data_list.append(batch_data)

                total += batch_data.size(0)
                assert sents_len == ([sents_len[0]] * len(sents_len))

        return batch_data_list

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
                batch_data, sents_len = self._to_tensor(batch_data, batch_first, device,min_len)
                batch_data_list.append(batch_data)
                batch_feat = torch.tensor(
                    np.array(batch_feat), dtype=torch.float, requires_grad=False, device=device)
                batch_feat_list.append(batch_feat)
                batch_domid = torch.tensor(
                    np.array(batch_domid), dtype=torch.long, requires_grad=False, device=device)
                batch_domid_list.append(batch_domid)
                # batch_label = torch.tensor(
                    # np.array(batch_label), dtype=torch.long, requires_grad=False, device=device)
                batch_label_list.append(batch_label)

                total += batch_data.size(0)
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
