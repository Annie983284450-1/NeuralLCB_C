#!/bin/bash
#SBATCH --job-name=NeuralGreedyV2_cp_septic                           # Job name
#SBATCH -A gts-bzhao94                             # Charge account
#SBATCH -N1 --gres=gpu:V100:1                       # Number of nodes and GPUs required
#SBATCH --mem-per-gpu=12G                           # Memory per gpu
#SBATCH -t30:00:00                                       # Duration of the job  
#SBATCH -qinferno                                   # QOS name
#SBATCH --output=/storage/home/hcoda1/6/azhou60/scratch/neuralcb_results/testSeptic250_trainSeptic2000_S6_ApproxNeuralLinLCBV2_cp[G=septic].log 
#SBATCH --mail-type=BEGIN,END,FAIL                  # Mail preferences
#SBATCH --mail-user=azhou60@gatech.edu            # e-mail address for notifications


module load anaconda3
conda activate neuralcbpy39
cd /storage/home/hcoda1/6/azhou60/p-bzhao94-0/NeuralLCB_C_phoniex
srun python tune_realworld.py --algo_groups ApproxNeuralLinLCBV2_cp --mode S6 --G septic --test 0 

 