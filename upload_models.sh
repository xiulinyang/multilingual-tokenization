#!/bin/bash

# Configurations
lang=$1
vocab=$2
repo_id=$3
start=1
end=10
step=1

#python models/rename_checkpoints.py -p models/${lang}_${vocab}_41

# Log into huggingface (optional if not done)
# huggingface-cli login --token $hf_token
huggingface-cli repo create $repo_id --yes
# Clone your repo
git clone https://huggingface.co/xiulinyang/$repo_id
cd $(basename "$repo_id")
cp /scratch/xiulyang/multilingual-tokenization/models/${lang}_${vocab}_41/* ./
git add .
git commit -m "Add checkpoint"
git push origin main


for checkpoint in $(seq $start $step $end)
do
    branch_name="epoch-${checkpoint}"
    echo "Pushing branch: $branch_name"

    # Create and checkout new branch
    git checkout -B $branch_name

    # Copy or move your checkpoint files into this directory
    # (assumes your checkpoints are saved somewhere already)
    cp -r /scratch/xiulyang/multilingual-tokenization/models/${lang}_${vocab}_41/epoch_$checkpoint/* ./

    # Stage, commit, push
    git add .
    git commit -m "Add checkpoint at epoch $checkpoint"
    git push origin $branch_name
    git checkout main

done

echo "All checkpoints pushed!"

