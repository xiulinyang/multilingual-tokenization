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

languages = args.language
vocab_size  = args.vocab_size
random_seed = args.random_seed

for lang in tqdm(languages):
    tokenizer = AutoTokenizer.from_pretrained(f'models/{lang}_{vocab_size}_{random_seed}')
    sents = Path(f'/scratch/xiulyang/multilingual-LM/data/multilingual/{lang}/test/{lang}.test').read_text().strip().split('\n')
    token_num =0
    tokenized_text =[]
    for sent in tqdm(sents):
        tokens = tokenizer.tokenize(sent)
        tokenized_text.extend(tokens)
    with open(f'freq_epoch/{lang}.json', 'w', encoding='utf-8',) as f:
        freq = Counter(tokenized_text)
        json.dump(freq, f,ensure_ascii=False)