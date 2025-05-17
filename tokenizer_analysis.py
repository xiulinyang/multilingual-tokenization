from transformers import AutoTokenizer, AutoModel
from pathlib import Path
import math
import json
from tqdm import tqdm
from collections import Counter
import argparse
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

args = parser.parse_args()

lang = args.language
vocab_size  = args.vocab_size
random_seed = args.random_seed

# languages = ['AR', 'DE', 'EN', 'TR', 'RU']
# vocab_sizes = ['5000', '10000' ,'20000', '30000', '50000']
# for lang in tqdm(languages):
#     for vocab_size in tqdm(vocab_sizes):
tokenizer = AutoTokenizer.from_pretrained(f'models/{lang}_{vocab_size}_41')
sents = Path(f'/scratch/xiulyang/multilingual-LM/data/multilingual/{lang}/test/{lang}.test').read_text().strip().split('\n')
token_num =0
tokenized_text =[]
for sent in tqdm(sents):
    tokens = tokenizer.tokenize(sent)
    tokenized_text.extend(tokens)
with open(f'freq_epoch/{lang}.json', 'w') as f:
    freq = Counter(tokenized_text)
    json.dump(freq, f,ensure_ascii = False, indent = 4)
