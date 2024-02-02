
from __future__ import absolute_import, division, print_function


import pdb
import argparse
import glob
import logging

import os
import pickle
import random

import numpy as np
import torch
import torch.nn.init as init
from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler, TensorDataset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange
from collections import defaultdict
from datetime import datetime
from .pytorch_transformers import (WEIGHTS_NAME, AdamW, WarmupLinearSchedule,
                                  BertConfig, BertForLatentConnector, BertTokenizer,
                                  GPT2Config, GPT2ForLatentConnector, GPT2Tokenizer,
                                  OpenAIGPTConfig, OpenAIGPTLMHeadModel, OpenAIGPTTokenizer,
                                  RobertaConfig, RobertaForMaskedLM, RobertaTokenizer)

logger = logging.getLogger(__name__)


MODEL_CLASSES = {
    'gpt2': (GPT2Config, GPT2ForLatentConnector, GPT2Tokenizer),
    'openai-gpt': (OpenAIGPTConfig, OpenAIGPTLMHeadModel, OpenAIGPTTokenizer),
    'bert': (BertConfig, BertForLatentConnector, BertTokenizer),
    'roberta': (RobertaConfig, RobertaForMaskedLM, RobertaTokenizer)
}

def configure_prevae(encoder_model_type,encoder_model_name_or_path,decoder_model_type,decoder_model_name_or_path,latent_size,save,load_weight,device):
    ## Encoder 
    encoder_config_class, encoder_model_class, encoder_tokenizer_class = MODEL_CLASSES[encoder_model_type]
    encoder_config = encoder_config_class.from_pretrained(encoder_model_name_or_path)
    tokenizer_encoder = encoder_tokenizer_class.from_pretrained(encoder_model_name_or_path, do_lower_case=True)
    # if .block_size <= 0:
    block_size = tokenizer_encoder.max_len_single_sentence  # Our input block size will be the max possible for the model
    block_size = min(block_size, tokenizer_encoder.max_len_single_sentence)
    model_encoder = encoder_model_class.from_pretrained(encoder_model_name_or_path, from_tf=bool('.ckpt' in encoder_model_name_or_path), config=encoder_config, latent_size=latent_size)

    #decoder
    decoder_config_class, decoder_model_class, decoder_tokenizer_class = MODEL_CLASSES[decoder_model_type]
    decoder_config = decoder_config_class.from_pretrained(decoder_model_name_or_path)
    tokenizer_decoder = decoder_tokenizer_class.from_pretrained(decoder_model_name_or_path, do_lower_case=True)
    # if .block_size <= 0:
    block_size = tokenizer_decoder.max_len_single_sentence  # Our input block size will be the max possible for the model
    block_size = min(block_size, tokenizer_decoder.max_len_single_sentence)
    
    latent_as_gpt_emb = True 
    latent_as_gpt_memory = True

    setattr(decoder_config, "latent_size", latent_size)
    model_decoder = decoder_model_class.from_pretrained(decoder_model_name_or_path, from_tf=bool('.ckpt' in decoder_model_name_or_path), config=decoder_config, latent_size=latent_size, latent_as_gpt_emb=latent_as_gpt_emb, latent_as_gpt_memory=latent_as_gpt_memory)
    
    # Chunyuan: Add Padding token to GPT2
    special_tokens_dict = {'pad_token': '<PAD>', 'bos_token': '<BOS>', 'eos_token': '<EOS>'}
    num_added_toks = tokenizer_decoder.add_special_tokens(special_tokens_dict)
    print('We have added', num_added_toks, 'tokens to GPT2')
    model_decoder.resize_token_embeddings(len(tokenizer_decoder))  # Notice: resize_token_embeddings expect to receive the full size of the new vocabulary, i.e. the length of the tokenizer.
    assert tokenizer_decoder.pad_token == '<PAD>'

    return model_encoder, model_decoder, tokenizer_encoder, tokenizer_decoder