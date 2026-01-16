#!/bin/bash -l

#SBATCH --account yshresth_aim
#SBATCH --output logs/%A.log
#SBATCH --partition gpu
#SBATCH --gres gpu:1
#SBATCH --gres-flags enforce-binding
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 8
#SBATCH --mem 16G
#SBATCH --time 15:00:00
#SBATCH --array=0-0

python extract-connectivity.py --attention-path "/scratch/noli1/graph_hallucination_detection/results/llava-med-v1.5-mistral-7b_train" --metadata-path "/scratch/noli1/graph_hallucination_detection/results/llava-med-v1.5-mistral-7b_train/metadata.pkl" --connectivity "mst" --output "./outputs/mst_train.pkl"
