from transformers import GPT2LMHeadModel, GPT2TokenizerFast
import pandas as pd

device = "cuda"
model_id = "gpt2"
model = GPT2LMHeadModel.from_pretrained(model_id).to(device)
tokenizer = GPT2TokenizerFast.from_pretrained(model_id)

def cal_ppl(target):
    target = target.dropna()
    text = target.content.tolist()
    text_str = "\n\n".join(text)
    word_list = text_str.split()
    # all_encodings = tokenizer(text_str, return_tensors="pt")

    import torch
    from tqdm import tqdm

    max_length = model.config.n_positions
    stride = 1000
    seq_len = len(word_list)

    nlls = []
    prev_end_loc = 0
    for begin_loc in range(0, seq_len, stride):
        end_loc = min(begin_loc + stride, seq_len)
        trg_len = end_loc - prev_end_loc  # may be different from stride on last loop
        input_seq = word_list[begin_loc:end_loc]
        encodings = tokenizer(" ".join(input_seq), return_tensors="pt")
        input_ids = encodings.input_ids[:,:1024].to(device)
        # input_ids = encodings.input_ids[:, begin_loc:end_loc].to(device)
        target_ids = input_ids.clone()
        target_ids[:, :-trg_len] = -100

        with torch.no_grad():
            outputs = model(input_ids, labels=target_ids)

            # loss is calculated using CrossEntropyLoss which averages over input tokens.
            # Multiply it with trg_len to get the summation instead of average.
            # We will take average over all the tokens to get the true average
            # in the last step of this example.
            neg_log_likelihood = outputs.loss * trg_len

        nlls.append(neg_log_likelihood)

        prev_end_loc = end_loc
        if end_loc == seq_len:
            break

    ppl = torch.exp(torch.stack(nlls).sum() / end_loc)
    return ppl.item()
    # print(ppl.item())