#!/bin/bash

CUDA_VISIBLE_DEVICES=2 python perplexities_exp.py AR 8680 41 bpc
CUDA_VISIBLE_DEVICES=2 python perplexities_exp.py TR 13222 41 bpc
CUDA_VISIBLE_DEVICES=2 python perplexities_exp.py RU 21042 41 bpc
CUDA_VISIBLE_DEVICES=2 python perplexities_exp.py DE 42180 41 bpc



CUDA_VISIBLE_DEVICES=2 python perplexities_exp.py AR 5000 41 bpc
CUDA_VISIBLE_DEVICES=2 python perplexities_exp.py AR 10000 41 bpc
CUDA_VISIBLE_DEVICES=2 python perplexities_exp.py AR 20000 41 bpc
CUDA_VISIBLE_DEVICES=2 python perplexities_exp.py AR 30000 41 bpc
CUDA_VISIBLE_DEVICES=2 python perplexities_exp.py AR 50000 41 bpc

CUDA_VISIBLE_DEVICES=2 python perplexities_exp.py DE 5000 41 bpc
CUDA_VISIBLE_DEVICES=2 python perplexities_exp.py DE 10000 41 bpc
CUDA_VISIBLE_DEVICES=2 python perplexities_exp.py DE 20000 41 bpc
CUDA_VISIBLE_DEVICES=2 python perplexities_exp.py DE 30000 41 bpc
CUDA_VISIBLE_DEVICES=2 python perplexities_exp.py DE 50000 41 bpc

CUDA_VISIBLE_DEVICES=2 python perplexities_exp.py EN 5000 41 bpc
CUDA_VISIBLE_DEVICES=2 python perplexities_exp.py EN 10000 41 bpc
CUDA_VISIBLE_DEVICES=2 python perplexities_exp.py EN 20000 41 bpc
CUDA_VISIBLE_DEVICES=2 python perplexities_exp.py EN 30000 41 bpc
CUDA_VISIBLE_DEVICES=2 python perplexities_exp.py EN 50000 41 bpc

CUDA_VISIBLE_DEVICES=2 python perplexities_exp.py TR 5000 41 bpc
CUDA_VISIBLE_DEVICES=2 python perplexities_exp.py TR 10000 41 bpc
CUDA_VISIBLE_DEVICES=2 python perplexities_exp.py TR 20000 41 bpc
CUDA_VISIBLE_DEVICES=2 python perplexities_exp.py TR 30000 41 bpc
CUDA_VISIBLE_DEVICES=2 python perplexities_exp.py TR 50000 41 bpc

CUDA_VISIBLE_DEVICES=2 python perplexities_exp.py RU 5000 41 bpc
CUDA_VISIBLE_DEVICES=2 python perplexities_exp.py RU 10000 41 bpc
CUDA_VISIBLE_DEVICES=2 python perplexities_exp.py RU 20000 41 bpc
CUDA_VISIBLE_DEVICES=2 python perplexities_exp.py RU 30000 41 bpc
CUDA_VISIBLE_DEVICES=2 python perplexities_exp.py RU 50000 41 bpc

