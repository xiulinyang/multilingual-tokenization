# perplexities.py
# Author: Julie Kallini

# For importing utils
import sys
sys.path.append("..")

from transformers import GPT2LMHeadModel, AutoTokenizer
from tqdm import tqdm
from glob import glob
import pandas as pd
import torch
import itertools
import argparse
import os



MAX_TRAINING_STEPS = 10000
CHECKPOINTS = list(range(0, MAX_TRAINING_STEPS+1, 200))


def create_attention_mask(token_lists):
    seq_length = max([len(i) for i in token_lists])
    batch_size = len(token_lists)
    mask = torch.full((batch_size, seq_length), 0)

    for i, tokens in enumerate(token_lists):
        mask[i, 0:len(tokens)] = 1

    return mask


def create_input_ids(lang, token_lists, pad_token_id):
    if lang=='ZH':
        pad_token_id = 0
    padded = zip(*itertools.zip_longest(*token_lists, fillvalue=pad_token_id))
    return torch.tensor(list(padded))


def get_perplexities(model, token_lists, pad_token_id, lang, device="cuda"):

    # Prepare data
    input_ids = create_input_ids(lang, token_lists, pad_token_id).to(device)
    labels = input_ids.clone()  # GPT-2 uses input as labels for CLM task
    attention_mask = create_attention_mask(token_lists).to(device)

    # Forward pass
    outputs = model(input_ids=input_ids, labels=labels,
                    attention_mask=attention_mask)

    # The "shifted" nature of labels in GPT-2 (next token prediction)
    # Shift logits, labels, and attention mask by one position
    shift_logits = outputs.logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    shift_attention_mask = attention_mask[..., 1:].contiguous()

    # Instantiate loss function with no reduction
    loss_fct = torch.nn.CrossEntropyLoss(reduction='none')

    # Calculate per-token loss
    loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1))

    # Reshape back to the original batch size and sequence length
    loss = loss.view(shift_labels.size())

    # Apply the attention mask - only calculate loss where mask is 1
    loss = loss * shift_attention_mask

    # Sum the loss over the sequence length, get per-example perplexity
    per_example_loss = loss.sum(dim=1) / shift_attention_mask.sum(dim=1)
    return torch.exp(per_example_loss).tolist()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        prog='Edge probing',
        description='Edge probing experiments')
    parser.add_argument('language',
                        default='all',
                        const='all',
                        nargs='?',
                        help='languages')
    parser.add_argument('vocab_size', help='Vocabulary size')
    parser.add_argument('random_seed', type=int, help="Random seed")


    # Get args
    args = parser.parse_args()
    vs = args.vocab_size
    random_seed = args.random_seed
    la = args.language
    # Get path to model
    model_path = f"models/{la}_{vs}_{random_seed}"
    models = glob(f"models/{la}_{vs}_{random_seed}/checkpoint-*")
    checkpoints = sorted([int(x.split('-')[-1]) for x in models])
    print(checkpoints)
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
    # Get perturbed test files
    test_files = [f"/scratch/xiulyang/multilingual-LM/data/multilingual/{la}/test/{la}.test"]

    # Iterate over data files to get perplexity data
    print("Sampling test data")
    text_sequences = []
    token_sequences =[]
    for test_file in test_files:
        print(test_file)

        # Get tokens from test file and subsample
        f = open(test_file, 'r')
        file_text_sequences = [l.strip() for l in f.readlines()]
        file_token_sequences = [tokenizer.encode(l.strip()) for l in file_text_sequences]
        text_sequences.extend(file_text_sequences)
        token_sequences.extend(file_token_sequences)
        print(text_sequences[:10])
        print(token_sequences[:10])
    # # For logging/debugging, include decoded sentence
    # test_sents = [gpt2_tokenizer.decode(
    #     toks) for toks in token_sequences]

    ppl_df = pd.DataFrame({
        "Sentences": text_sequences,
    })

    BATCH_SIZE = 8
    device = "cuda"
    for i, ckpt in enumerate(checkpoints):
        print(f"Epoch: {i} (ckpt: {ckpt})")

        # Load model
        model = GPT2LMHeadModel.from_pretrained(
        model_path + '/checkpoint-'+ str(ckpt)).to(device)
        print("Tokenizer vocab size:", len(tokenizer))
        print("Model vocab size:", model.config.vocab_size)
        # Get perplexities
        perplexities = []
        for i in tqdm(range(0, len(token_sequences), BATCH_SIZE)):
            batch = token_sequences[i:i+BATCH_SIZE]
            ppls = get_perplexities(
                model, batch, tokenizer.eos_token_id, la)
            perplexities.extend(ppls)

        # Add ppls to df
        ppl_df[f'Perplexities (ckpt {ckpt})'] = perplexities

    # Write results to CSV
    directory = f"perplexity_results/{la}_{vs}"
    file = directory + \
           f"/{la}_{vs}_seed{args.random_seed}.csv"

    if not os.path.exists(directory):
        os.makedirs(directory)
    vs = str(vs)
    print(f"Writing results to CSV: {file}")
    ppl_df.to_csv(file)
