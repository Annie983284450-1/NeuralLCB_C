#!/bin/bash
#SBATCH --job-name=NeuralGreedyV2_cp_septic                           # Job name
#SBATCH -A gts-bzhao94                             # Charge account
#SBATCH -N1 --gres=gpu:V100:1                       # Number of nodes and GPUs required
#SBATCH --mem-per-gpu=12G                           # Memory per gpu
#SBATCH -t30:00:00                                       # Duration of the job  
#SBATCH -qinferno                                   # QOS name
#SBATCH --output=/storage/home/hcoda1/6/azhou60/scratch/neuralcb_results/testS1_ApproxNeuraLCB_cp[G=septic].log
#SBATCH --mail-type=BEGIN,END,FAIL                  # Mail preferences
#SBATCH --mail-user=azhou60@gatech.edu            # e-mail address for notifications


module load anaconda3
conda activate neuralcbpy39
cd $SLURM_SUBMIT_DIR                        # Change to working directory created in $HOME
srun python tune_realworld.py --algo_groups NeuralGreedyV2_cp --mode S7 --G septic --test 0 

 