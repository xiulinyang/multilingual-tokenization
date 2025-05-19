#!/bin/bash

# Configurations
lang=$1
vocab=$2
seed=$3
python models/rename_checkpoints.py -p models/${lang}_${vocab}_${seed}
