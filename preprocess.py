# Copyright (c) 2020-present, Royal Bank of Canada.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import json
import numpy as np
import argparse
import os
import pandas as pd
import config
import codecs
import json
import torch
from transformers import BertTokenizer, BertModel

def preprocess(split='train', unk='_UNK'):
    input_file = "data/yelp_data/yelp.{}.txt".format(split)
    output_file = "data/yelp_data/_{}.txt".format(split)
    with open(input_file) as f_in:
        f_out = open(output_file, "w")
        for content in f_in.readlines():
            content = content.split('\t')[1]
            content = content.replace(unk, UNK_TOKEN)
            content = content.replace('-lrb-', '(')
            content = content.replace('-rrb-', ')')
            content = content.replace('-lsb-', '[')
            content = content.replace('-rsb-', ']')
            content = content.replace('-lcb-', '{')
            content = content.replace('-rcb-', "}")
            f_out.write(content)

def get_bert_embeds(in_file,out_file):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained("bert-base-uncased")
    embeds = []
    bsz = 64
    bsz_id,line_id = 0,0
    file_len = len(open(in_file).readlines())
    with open(in_file) as f:
        batch_lines=[]
        for line in f:#batch_size
            text = line.split("\t")[1].strip()
            batch_lines.append(text)
            line_id += 1
            bsz_id += 1
            if bsz_id == bsz or line_id == file_len:
                encoded_input = tokenizer(batch_lines, return_tensors='pt',padding=True)
                output = model(**encoded_input)
                # torch.cuda.empty_cache()
                print("%d of 4000 samples"%(line_id))
                bsz_id=0
                embeds.append(output[1].detach().cpu().numpy()) 
        embeds = np.concatenate(embeds)
        assert embeds.shape[0]==file_len
        np.save(out_file, embeds)
        print("Save BERT [CLS] Embeddings")

def get_glove_embeds(in_file, out_file):
    glove_file = "../glove.840B.300d.txt"
    word_vec = {}
    with open(glove_file) as f:
        for line in f:
            word, vec = line.split(' ', 1)
            word_vec[word] = np.fromstring(vec, sep=' ')

    embeds = []
    with open(in_file) as f:
        for line in f:
            tokens = line.strip().split()
            vec = np.zeros(300, dtype=np.float32)
            sent_len = 0
            for token in tokens:
                if token in word_vec:
                    vec += word_vec[token]
                    sent_len += 1
            if sent_len > 0:
                vec = np.true_divide(vec, sent_len)
            vec = vec.reshape(1, 300)
            embeds.append(vec)
        embeds = np.concatenate(embeds)
        np.save(out_file, embeds)


def concat_files(pth0, pth1, outpth, with_label=True):
    with open(outpth, "w") as f_out:
        with open(pth0, errors='ignore') as f0:
            for line in f0.readlines():
                if with_label:
                    f_out.write("0\t")
                f_out.write(line.strip() + "\n")
        with open(pth1, errors='ignore') as f1:
            for line in f1.readlines():
                if with_label:
                    f_out.write("1\t")
                f_out.write(line.strip() + "\n")

def flip_files(pth, outpth, with_label=True):
    with open(outpth, "w") as f_out:
        with open(pth, errors='ignore') as f:
            for line in f.readlines():
                label, content = line.strip().split("\t")
                if with_label:
                    f_out.write("%d\t" % (1 - int(label)))
                f_out.write(content + "\n")

def read_data(data_name):
    # LOAD DATA
    domain_dict = {"imdb":0,"yelp_dast":1,"amazon":2,"yahoo":3}
    for split in ["train","dev","test"]:
        reader = codecs.open("../data/%s/%s.txt" %(data_name,split))
        outfile = open("../data/%s/%s_data.txt" %(data_name,split),"w+")
        line_id = 0
        while True:
            string_ = reader.readline()
            line_id += 1
            if not string_: break
            dict_example = json.loads(string_)
            review = dict_example["review"]
            score = dict_example["score"]
            domain_id = domain_dict[data_name]
            outfile.write(str(score)+"\t"+review+"\t"+str(domain_id)+"\n")
        outfile.close()
        print("Processing %s Dataset %s Split %d samples"%(data_name,split,line_id))

def main(args):
    data_pth = "../data/%s" % args.data_name
    res_pth = "results/%s" % args.data_name
    # read_data(args.data_name)
    #save as the train_data.txt, dev_data.txt, test_data.txt
    for split in ["train", "dev", "test"]:
        # pth0 = "sentiment.%s.0" % split
        # pth1 = "sentiment.%s.1" % split
        outpth = "%s_data.txt" % split
        # _outpth = "_%s_data.txt" % split
        # pth0 = os.path.join(data_pth, pth0)
        # pth1 = os.path.join(data_pth, pth1)
        outpth = os.path.join(data_pth, outpth)
        # _outpth = os.path.join(data_pth, _outpth)
        # concat_files(pth0, pth1, outpth)
        # concat_files(pth0, pth1, _outpth, False)
        print("Generating feat.npy for %s"%split)
        fin = outpth
        fout = os.path.join(data_pth, "%s_bert.npy" % split)
        # get_glove_embeds(fin, fout)
        get_bert_embeds(fin,fout)

def add_args(parser):
    parser.add_argument('--data_name', type=str, default='imdb')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_args(parser)
    args = parser.parse_args()
    main(args)
