#!/bin/bash
#SBATCH --job-name=cal_penetration
#SBATCH --output=logs/cal_pen_%a.out
#SBATCH --error=logs/cal_pen_%a.err
#SBATCH --array=0-255             
#SBATCH --cpus-per-task=1        
#SBATCH --mem=3G                
#SBATCH --time=1-00:00:00

source ~/.bashrc
conda activate pico

mkdir -p logs
python src/evaluation/interpenetration.py