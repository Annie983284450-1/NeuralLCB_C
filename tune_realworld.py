import os
import subprocess
import time
import argparse
import random
import glob
import numpy as np
import platform
 



parser = argparse.ArgumentParser()

 
# parser.add_argument('--task', type=str, default='run_exps', choices=['run_exps','collect_results'])
parser.add_argument('--task', type=str, default='run_exps', choices=['run_exps'])
parser.add_argument('--data_types', nargs='+', type=str, default=['sepsis'])
parser.add_argument('--algo_groups', nargs='+', type=str, default=['ApproxNeuraLCB_cp'])
parser.add_argument('--policies', nargs='+', type=str, default=['eps-greedy'])
parser.add_argument('--num_sim', type=int, default=1)
parser.add_argument('--models_per_gpu', type=int, default=3)
parser.add_argument('--gpus', nargs='+', type=int, default=[0], help='gpus indices used for multi_gpu')

parser.add_argument('--result_dir', type=str, default='results/stock_d=21_a=8_pi=eps-greedy0.1_std=0.1', help='result directory for collect_results()')
args = parser.parse_args()



def multi_gpu_launcher(commands, gpus=None, models_per_gpu=1):
    """
    Launch commands on the local machine or server, using all GPUs in parallel.
    Works on Windows (1 GPU), Linux, and macOS with multiple GPUs.
    """
    system_platform = platform.system()
    
    # Detect GPUs if not specified
    if gpus is None:
        if system_platform == "Windows" or system_platform == "Linux":
            gpus = detect_gpus()
        elif system_platform == "Darwin":
            gpus = []  # macOS doesn't have CUDA by default

    if not gpus:
        raise RuntimeError("No GPUs detected or specified.")

    procs = [None] * (len(gpus) * models_per_gpu)

    while len(commands) > 0:
        for i, proc in enumerate(procs):
            gpu_idx = gpus[i % len(gpus)]
            if (proc is None) or (proc.poll() is not None):
                cmd = commands.pop(0)

                if system_platform == "Linux" or system_platform == "Darwin":  # Linux/macOS
                    env = os.environ.copy()
                    env["CUDA_VISIBLE_DEVICES"] = str(gpu_idx)
                    new_proc = subprocess.Popen(cmd, shell=True, env=env)
                
                elif system_platform == "Windows":
                    # Windows specific case
                    if len(gpus) == 1:
                        # Only one GPU on Windows, run all tasks on it
                        new_proc = subprocess.Popen(cmd, shell=True, creationflags=subprocess.CREATE_NEW_PROCESS_GROUP)
                    else:
                        if get_gpu_utilization(gpu_idx) < 50:
                            new_proc = subprocess.Popen(cmd, shell=True, creationflags=subprocess.CREATE_NEW_PROCESS_GROUP)
                        else:
                            continue  # Skip if GPU is too busy

                procs[i] = new_proc
                break
        time.sleep(1)

    # Wait for the last few tasks to finish before returning
    for p in procs:
        if p is not None:
            p.wait()


def detect_gpus():
    """
    Detect available GPUs using nvidia-smi (if available).
    """
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
            stdout=subprocess.PIPE, text=True, check=True
        )
        gpu_indices = result.stdout.strip().split("\n")
        return [int(idx) for idx in gpu_indices]
    except Exception as e:
        print(f"Error detecting GPUs: {e}")
        return [0]  # Default to GPU 0 if detection fails or only one GPU present


def get_gpu_utilization(gpu_idx):
    """
    Query the current GPU utilization using nvidia-smi.
    """
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,nounits,noheader"],
            stdout=subprocess.PIPE, text=True, check=True
        )
        utilization = result.stdout.splitlines()[gpu_idx]
        return int(utilization.strip())
    except Exception as e:
        print(f"Error querying GPU {gpu_idx}: {e}")
        return 100  # Assume full utilization if query fails


def multi_gpu_launcher_linux_mac(commands,gpus,models_per_gpu):
    """
    Launch commands on the local machine, using all GPUs in parallel.
    """
    procs = [None]*len(gpus)*models_per_gpu

    while len(commands) > 0:
        for i,proc in enumerate(procs):
            gpu_idx = gpus[i % len(gpus)]
            if (proc is None) or (proc.poll() is not None):
                # Nothing is running on this index; launch a command.
                cmd = commands.pop(0)
                new_proc = subprocess.Popen(
                    f'CUDA_VISIBLE_DEVICES={gpu_idx} {cmd}', shell=True)
                procs[i] = new_proc
                break
        time.sleep(1)

    # Wait for the last few tasks to finish before returning
    for p in procs:
        if p is not None:
            p.wait()


 

 
# hyper_mode = 'best' # ['full', 'best']
# if hyper_mode == 'full':
#     # Grid search space: used for grid search in the paper
#     lr_space = [1e-4,1e-3]
#     train_mode_space = [(1,1,1),(50,100,-1)]
#     beta_space = [0.01, 0.05, 1,5,10]
#     rbfsigma_space = [1] #[0.1, 1,10]
# elif hyper_mode == 'best':
#     # The best searc sapce inferred fro prior experiments 
#     lr_space = [1e-4]
#     train_mode_space = [(1,1,1)]
#     beta_space = [1]
#     rbfsigma_space = [10] #10 for mnist, 0.1 for mushroom
 
# def create_commands(data_type='mushroom', algo_group='approx-neural', num_sim=3, policy='eps-greedy'):
#     commands = []
#     if algo_group == 'approx-neural':
#         for lr in lr_space:
#             for batch_size,num_steps,buffer_s in train_mode_space:
#                 for beta in beta_space:
#                     commands.append('python realworld_main.py --data_type {} --algo_group {} --num_sim {} --batch_size {} --num_steps {} --buffer_s {} --beta {} --lr {} --policy {}'.format(data_type,algo_group,num_sim,batch_size,num_steps,buffer_s,beta,lr,policy))

#     elif algo_group == 'neurallinlcb':
#         for beta in beta_space:
#             commands.append('python realworld_main.py --data_type {} --algo_group {} --num_sim {}  --beta {} --policy {}'.format(data_type,algo_group,num_sim,beta,policy))

    
#     elif algo_group == 'neural-greedy':
#         for lr in lr_space:
#             for batch_size,num_steps,buffer_s in train_mode_space:
#                 commands.append('python realworld_main.py --data_type {} --algo_group {} --num_sim {} --batch_size {} --num_steps {} --buffer_s {} --lr {} --policy {}'.format(data_type,algo_group,num_sim,batch_size,num_steps,buffer_s,lr,policy))

#     elif algo_group == 'kern':
#         for rbf_sigma in rbfsigma_space:
#             commands.append('python realworld_main.py --data_type {} --algo_group {} --num_sim {} --rbf_sigma {} --policy {}'.format(data_type,algo_group,num_sim,rbf_sigma,policy))

#     elif algo_group == 'baseline': # no tuning 
#         commands.append('python realworld_main.py --data_type {} --algo_group {} --num_sim {} --policy {}'.format(data_type,algo_group,num_sim,policy))

#     else:
#         raise NotImplementedError

#     return commands
def create_commands(data_type='sepsis', algo_group='ApproxNeuraLCB_cp', num_sim=1, policy='eps-greedy'):
    # hyper_mode = 'best' # ['full', 'best']
    hyper_mode = 'full' # ['full', 'best']
    if hyper_mode == 'full':
        # Grid search space: used for grid search in the paper
        lr_space = [1e-4,1e-3]
        train_mode_space = [(1,1,1),(50,100,-1)]
        beta_space = [0.01, 0.05, 1, 5,10] #[0.01, 0.05, 1,5,10]
        rbfsigma_space = [1] #[0.1, 1,10]
    elif hyper_mode == 'best':
        # The best searc sapce inferred fro prior experiments 
        lr_space = [1e-4]
        train_mode_space = [(1,1,1)]
        beta_space = [1]
        rbfsigma_space = [10] #10 for mnist, 0.1 for mushroom
    commands = []
    if algo_group == 'ApproxNeuraLCB_cp':
        for lr in lr_space:
            for batch_size,num_steps,buffer_s in train_mode_space:
                for beta in beta_space:
                    commands.append('python realworld_main.py --data_type {} --algo_group {} --num_sim {} --batch_size {} --num_steps {} --buffer_s {} --beta {} --lr {} --policy {}'.format(data_type,algo_group,num_sim,batch_size,num_steps,buffer_s,beta,lr,policy))
                    # commands.append('python realworld_main.py --data_type {} --algo_group {} --num_sim {}  --beta {} --policy {}'.format(data_type,algo_group,num_sim,beta,policy))
                    # commands.append('python realworld_main.py --data_type {} --algo_group {} --num_sim {} --batch_size {} --num_steps {} --buffer_s {} --lr {} --policy {}'.format(data_type,algo_group,num_sim,batch_size,num_steps,buffer_s,lr,policy))

    elif algo_group == 'ApproxNeuralLinLCBV2_cp' or 'ApproxNeuralLinLCBJointModel_cp':
        for beta in beta_space:
            commands.append('python realworld_main.py --data_type {} --algo_group {} --num_sim {}  --beta {} --policy {}'.format(data_type,algo_group,num_sim,beta,policy))

    
    elif algo_group == 'NeuralGreedyV2_cp':
        for lr in lr_space:
            for batch_size,num_steps,buffer_s in train_mode_space:
                commands.append('python realworld_main.py --data_type {} --algo_group {} --num_sim {} --batch_size {} --num_steps {} --buffer_s {} --lr {} --policy {}'.format(data_type,algo_group,num_sim,batch_size,num_steps,buffer_s,lr,policy))

    # elif algo_group == 'kern':
    #     for rbf_sigma in rbfsigma_space:
    #         commands.append('python realworld_main.py --data_type {} --algo_group {} --num_sim {} --rbf_sigma {} --policy {}'.format(data_type,algo_group,num_sim,rbf_sigma,policy))

    # elif algo_group == 'baseline': # no tuning 
    #     commands.append('python realworld_main.py --data_type {} --algo_group {} --num_sim {} --policy {}'.format(data_type,algo_group,num_sim,policy))

    else:
        raise NotImplementedError

    return commands

def run_exps():
    commands = []
    for data_type in args.data_types:
        for algo_group in args.algo_groups:
            for policy in args.policies:
                commands += create_commands(data_type, algo_group, args.num_sim, policy)
    random.shuffle(commands)
    multi_gpu_launcher(commands, args.gpus, args.models_per_gpu)

# def collect_results():
#     filenames = glob.glob(os.path.join(args.result_dir,"*.npz"))
#     results = {}
#     for filename in filenames:
#         k = np.load(filename)
#         regret = k['arr_0'][:,1,:]
#         regret = np.min(regret,1) # best regret of a run
#         regret = np.mean(regret)
#         results[filename] = regret
    
#     filenames.sort(key=lambda x: results[x])
    
#     for filename in filenames:
#         print('{}:   {}'.format(filename,results[filename]))

if __name__ == '__main__':
    eval(args.task)()
