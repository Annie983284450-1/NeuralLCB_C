# !/bin/bash
# SBATCH --job-name=neuralcbcptest_test5_train20                  # job name
# SBATCH -A gts-bzhao94                   # account to which job is charged, ex: GT-gburdell3
# SBATCH -N1 --ntasks-per-node=1    	     # number of nodes and cores per node required
# SBATCH --mem-per-cpu=2G                 # memory per core
# SBATCH -t60                             # duration of the job (ex: 10 min)
# SBATCH -oReport-%j.out                  # output file name

cd $SLURM_SUBMIT_DIR

module load anaconda3
conda activate jax_tf28

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

# cd $HOME/slurm_gpu_example                          # Change to working directory created in $HOME

module load tensorflow-gpu/2.9.0                    # Load module dependencies
srun python $TENSORFLOWGPUROOT/testgpu.py gpu 1000  # Run test example
