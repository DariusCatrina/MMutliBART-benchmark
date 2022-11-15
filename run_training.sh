#!/bin/sh
#SBATCH --partition=bhuwan
#SBATCH --mem-per-cpu=128G
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
 
torchrun --nnodes=1 --nproc_per_node=1 train_model.py --seq_len=4096 --bsz=5 --dataset_name=gov_report --number_of_decoders=1 --bart_seq_len=1024
