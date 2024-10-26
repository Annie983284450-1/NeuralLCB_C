python realworld_main.py

#!/bin/bash
#SBATCH -JGPUTest.                                  # Job name
#SBATCH -A gts-bzhao94                             # Charge account
#SBATCH -N1 --gres=gpu:V100:1                       # Number of nodes and GPUs required
#SBATCH --mem-per-gpu=12G                           # Memory per gpu
#SBATCH -t15                                        # Duration of the job (Ex: 15 mins)
#SBATCH -qinferno                                   # QOS name
#SBATCH -oReport-%j.out                             # Combined output and error messages file
#SBATCH --mail-type=BEGIN,END,FAIL                  # Mail preferences
#SBATCH --mail-user=azhou60@gatech.edu            # e-mail address for notifications

# cd $SLURM_SUBMIT_DIR                        # Change to working directory created in $HOME

module load tensorflow-gpu/2.9.0                    # Load module dependencies
srun python $TENSORFLOWGPUROOT/testgpu.py gpu 1000  # Run test example
