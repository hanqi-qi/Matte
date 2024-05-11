# Copyright (c) 2020-present, Royal Bank of Canada.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
CUDA_VISIBLE_DEVICES=0

import pandas as pd
import argparse
from utils.bleu import compute_bleu,calculate_gscore,calculate_bleu
from nltk.translate.bleu_score import corpus_bleu, sentence_bleu, SmoothingFunction
smoothie = SmoothingFunction().method1
from utils.diversity import distinct_n_corpus_level,distinct_n_sentence_level
from utils.text_utils_our import MonoTextData
import torch
from classifier import CNNClassifier, evaluate_func
from nltk.translate.bleu_score import corpus_bleu, sentence_bleu, SmoothingFunction
import nltk.translate.gleu_score as gleu
import os
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
import pandas as pd
from utils.ppl import cal_ppl
# from ctc_score import StyleTransferScorer
import pandas as pd

smoothie = SmoothingFunction().method1

def main(args):
    data_pth ="/mnt/Data3/hanqiyan/UDA/real_world/data/"
    train_pth = os.path.join(data_pth, "train_all_data.txt")
    train_data = MonoTextData(train_pth,n_domains=1)
    vocab = train_data.vocab
    # source_pth = os.path.join(data_pth, "test_data.txt")
    source_pth = "/mnt/Data3/hanqiyan/style_transfer_baseline/checkpoint/model-yelp-glove-w0domid-cpvae-useflow-True-seed-39-allloss/20230715-193445/yelp_reference_results_flip.txt"
    target_pth = "/mnt/Data3/hanqiyan/style_transfer_baseline/checkpoint/model-yelp-glove-w0domid-cpvae-useflow-True-seed-39-allloss/20230715-193445/yelp_transfer_results_flip.txt"
    """"
    [CPVAE]

    [OptimusSNLi]
        "/mnt/Data3/hanqiyan/Optimus/unsupervised_transfer/yelp_transfer_step_3.txt"
    [BetaVAE]
        "/mnt/Data3/hanqiyan/CP-VAE/checkpoint/baseline-all_domains/20230401-143929/yahoo_generated_text_1.txt"
    [jointrain]
        "
    [indomain]
        "/mnt/Data3/hanqiyan/style_transfer_baseline/checkpoint/model-amazon-glove-w0domid-inDomain/20230329-160303/amazon_transfer_results_flip_InDomain.txt"
    # target_pth = "/mnt/Data3/hanqiyan/style_transfer_baseline/checkpoint/cpvae_pretrain-DoCoGen_review-glove/20230129-164731/imdb_transfer_results_0206.txt"
    """
    data_name = target_pth.split("/")[-1].split("_")[0]
    print("Evaluating the %s Dataet"%data_name)
    eval_data = MonoTextData(target_pth,n_domains=1,vocab=vocab)
    source = pd.read_csv(source_pth, names=['label', 'content'], sep='\t')
    target = pd.read_csv(target_pth, names=['label', 'content'], sep='\t')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Classification Accuracy
    model = CNNClassifier(len(vocab), 300, [1, 2, 3, 4, 5], 500, 0.5).to(device)
    model.load_state_dict(torch.load("/mnt/Data3/hanqiyan/style_transfer_baseline/checkpoint/yelp-classifier-pure.pt"))
    model.eval()
    eval_data,_, eval_label = eval_data.create_data_batch_domids(16, device, batch_first=True)
    acc, confid_list= evaluate_func(model, eval_data, eval_label)
    acc = 100*acc
    print("Acc: %.2f" % acc), confid_list

    # # # BLEU Score & ctc Score
    total_bleu = 0.0
    # scorer = StyleTransferScorer(align='E-roberta')
    sources = []
    targets = []
    source_sents = []
    # ctc_score = []
    confid = []
    for i in range(source.shape[0]):
        s = source.content[i].split()
        try:
            t = target.content[i].split()
            sources.append([s])
            source_sents.append(s)
            targets.append(t)
            confid.append(confid_list[i])
            # ctc_score.append(scorer.score(input_sent=" ".join(s), hypo=" ".join(t), aspect='preservation'))
        except:
            pass

    assert len(confid)==len(sources)==len(targets)
    bleu  = corpus_bleu(sources,targets)
    total_bleu = bleu*100
    print("Bleu: %.2f" % total_bleu)
    gscore = calculate_gscore(sources,targets,confid)
    # results = compute_bleu(reference_corpus=sources, translation_corpus=targets, acc_list=confid)

    print("G-score:%.2f"%(100*gscore))
    # print("CTC Score %.3f"%(sum(ctc_score)/len(ctc_score)))

    # # #PPL
    source_ppl = cal_ppl(source)
    target_ppl = cal_ppl(target)
    print("Reference PPL is %.2f, Transferred PPL is %.2f"%(source_ppl,target_ppl))

    #Diversity
    long_tar, long_src = [],[]
    for target, source in zip(targets,source_sents):
        long_tar.extend(target)
        long_src.extend(source)

    
    diversity = 100*distinct_n_sentence_level(long_tar,2)
    print("The diversity of 2-grams in Target is %.4f"%diversity)
    diversity = 100*distinct_n_sentence_level(long_src,2)
    print("The diversity of 2-grams in Source is %.4f"%diversity)

def add_args(parser):
    parser.add_argument('--data_name', type=str, default='DoCoGen_review')
    parser.add_argument('--target_path', type=str)
    parser.add_argument('--n_domains', default=6, type=int,
                        help='if use multi-domain dataset')  



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_args(parser)
    args = parser.parse_args()

    main(args)
