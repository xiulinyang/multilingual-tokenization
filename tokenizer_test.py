from torch.hub import ENV_GITHUB_TOKEN
from transformers import AutoTokenizer, AutoModel
from pathlib import Path
import math
from tqdm import tqdm
import argparse


parser = argparse.ArgumentParser(description='calculate tcw for different tokenizer')

parser.add_argument('-l', '--lang', help='language, e.g., EN, TR')
parser.add_argument('-v', '--vocab_size', help='vocab size')
args = parser.parse_args()
lang = args.lang
vocab_size = args.vocab_size

# for language in tqdm(TOKENIZER_DICT):
tokenizer = AutoTokenizer.from_pretrained(f'models/{lang}_{vocab_size}_41')
sents = Path(f'/scratch/xiulyang/multilingual-LM/data/multilingual/{lang}/test/{lang}.test').read_text().strip().split('\n')
token_num =0
bits_num =0
character_num =0
for sent in tqdm(sents):
    tokens = tokenizer.tokenize(sent)
    token_num += len(tokens)
    bits_num+=len(sent.encode('utf-8'))
    character_num+=len(sent)

with
average =token_num/len(sents)
print(lang, average)
print(token_num)
