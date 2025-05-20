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

languages = ['AR', 'DE', 'EN', 'TR', 'RU']
vocab_sizes = ['200', '5000', '10000' ,'20000', '30000', '50000']
with open('counts.tsv', 'w') as counts:
    counts.write('language\tvocab_size\tcorpus_token_counts\tbit_counts\tcharacter_counts\n')
    for lang in tqdm(languages):
        for vocab_size in tqdm(vocab_sizes):
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

            counts.write(f'{lang}\t{vocab_size}\t{token_num}\t{bits_num}\t{character_num}\n')


# average =token_num/len(sents)
# print(lang, average)
# print(token_num)
