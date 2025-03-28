import os
import subprocess
import time
import argparse
import random
import glob
import numpy as np
import platform
import sys



parser = argparse.ArgumentParser()

 
# parser.add_argument('--task', type=str, default='run_exps', choices=['run_exps','collect_results'])
parser.add_argument('--task', type=str, default='run_exps', choices=['run_exps'])
parser.add_argument('--data_types', nargs='+', type=str, default=['sepsis'])
parser.add_argument('--algo_groups', nargs='+', type=str, default=['ApproxNeuraLCB_cp'])
parser.add_argument('--policies', nargs='+', type=str, default=['eps-greedy'])
parser.add_argument('--num_sim', type=int, default=1)
parser.add_argument('--models_per_gpu', type=int, default=1)
parser.add_argument('--gpus', nargs='+', type=int, default=[0], help='gpus indices used for multi_gpu')

 
parser.add_argument('--mode', type=str, default=None, help='hyper mode')
parser.add_argument('--G', type=str, default='septic', help='data group')
parser.add_argument('--test', type=int, default=0, help='if test?')
parser.add_argument('--test_freq', type=int, default=100, help='test_freq')




args = parser.parse_args()



def test_multi_gpu_launcher(commands, gpus=None, models_per_gpu=3):
    """
    Launch commands on the local machine or server, using all GPUs in parallel.
    Works on Windows (1 GPU), Linux, and macOS with multiple GPUs.
    """
    system_platform = platform.system()
    
    # Detect GPUs if not specified
    if gpus is None:
        if system_platform == "Windows" or system_platform == "Linux":
            gpus = detect_gpus()
            print(f'{gpus} GPU detected!!!')
        elif system_platform == "Darwin":
            gpus = []  # macOS doesn't have CUDA by default

    if not gpus:
        raise RuntimeError("No GPUs detected or specified.")



def multi_gpu_launcher(commands, gpus=None, models_per_gpu=2):
    """
    Launch commands on the local machine or server, using all GPUs in parallel.
    Works on Windows (1 GPU), Linux, and macOS with multiple GPUs.
    """
    system_platform = platform.system()
    
    # Detect GPUs if not specified
    if gpus is None:
        if system_platform == "Windows" or system_platform == "Linux":
            gpus = detect_gpus()
            print(f'{gpus} GPU detected!!!')
        elif system_platform == "Darwin":
            gpus = []  # macOS doesn't have CUDA by default

    if not gpus:
        raise RuntimeError("No GPUs detected or specified.")

    procs = [None] * (len(gpus) * models_per_gpu)

    # Wait for the last few tasks to finish before returning
    for p in procs:
        if p is not None:
            p.wait()

    while len(commands) > 0:
        for i, proc in enumerate(procs):
            gpu_idx = gpus[i % len(gpus)]
            # if get_gpu_utilization(gpu_idx) < 50:
            if (proc is None) or (proc.poll() is not None):
                cmd = commands.pop(0)

                if system_platform == "Linux" or system_platform == "Darwin":  # Linux/macOS
                    env = os.environ.copy()
                    env["CUDA_VISIBLE_DEVICES"] = str(gpu_idx)
                    new_proc = subprocess.Popen(cmd, shell=True, env=env)
                    # new_proc = subprocess.Popen(
                    #     f'CUDA_VISIBLE_DEVICES={gpu_idx} {cmd}', shell=True)
                    # new_proc = subprocess.Popen(f'srun --gres=gpu:{gpu_idx} {cmd}', shell=True)  
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
    
  
 
def create_commands(data_type='sepsis', algo_group='ApproxNeuraLCB_cp', num_sim=1, policy='eps-greedy',hyper_mode = None, group = 'septic', test = False, test_freq = 100):
    # test = False
    # test = True

    # hyper_mode = 'A2:beta_noise005_tune'
    # group = 'septic'
    if hyper_mode == 'full':
        # Grid search space: used for grid search in the paper
        lr_space = [1e-4,1e-3]
        train_mode_space = [(1,1,1),(50,100,-1)]
        # beta_space = [0.01, 0.05, 1, 5,10] #[0.01, 0.05, 1,5,10]
        beta_space = [0.01, 0.05, 1, 5,10]
        rbfsigma_space = [1] #[0.1, 1,10]
        noise_std_space = [0.05, 0.1]
        cpr = 0.1
    elif hyper_mode == 'best':
        # The best searc sapce inferred fro prior experiments 
        lr_space = [1e-4]
        train_mode_space = [(1,1,1)]
        beta_space = [1]
        rbfsigma_space = [10] #10 for mnist, 0.1 for mushroom 
        cpr = 0.1
    elif hyper_mode == 'X0':
        lr_space = [0.0001]
        train_mode_space = [(32,100,-1)]
        beta_space = [0.01]
        noise_std_space = [0.1]
        cpr = 0.5
    elif hyper_mode == 'X1':
        lr_space = [0.0001]
        train_mode_space = [(32,100,-1)]
        beta_space = [0.01]
        noise_std_space = [0.1]
        cpr = 0.1
    elif hyper_mode == 'X2':
        lr_space = [0.0001]
        train_mode_space = [(32,100,-1)]
        beta_space = [0.1]
        noise_std_space = [0.1]
        cpr = 0.5
    elif hyper_mode == 'X3':
        lr_space = [0.0001]
        train_mode_space = [(32,100,-1)]
        beta_space = [0.1]
        noise_std_space = [0.1]
        cpr = 0.1
    elif hyper_mode == 'X4':
        lr_space = [0.0001]
        train_mode_space = [(32,100,-1)]
        beta_space = [0.05]
        noise_std_space = [0.1]
        cpr = 0.5
    elif hyper_mode == 'X5':
        lr_space = [0.0001]
        train_mode_space = [(32,100,-1)]
        beta_space = [0.05]
        noise_std_space = [0.1]
        cpr = 0.1
    elif hyper_mode == 'X6':
        lr_space = [0.0001]
        train_mode_space = [(32,100,-1)]
        beta_space = [0.5]
        noise_std_space = [0.1]
        cpr = 0.5
    elif hyper_mode == 'X7':
        lr_space = [0.0001]
        train_mode_space = [(32,100,-1)]
        beta_space = [0.5]
        noise_std_space = [0.1]
        cpr = 0.1
    elif hyper_mode == 'X8':
        lr_space = [0.0001]
        train_mode_space = [(32,100,-1)]
        beta_space = [1.0]
        noise_std_space = [0.1]
        cpr = 0.5
    elif hyper_mode == 'X9':
        lr_space = [0.0001]
        train_mode_space = [(32,100,-1)]
        beta_space = [1.0]
        noise_std_space = [0.1]
        cpr = 0.1
    elif hyper_mode == 'X10':
        lr_space = [0.0001]
        train_mode_space = [(32,100,-1)]
        beta_space = [5.0]
        noise_std_space = [0.1]
        cpr = 0.5
    elif hyper_mode == 'X11':
        lr_space = [0.0001]
        train_mode_space = [(32,100,-1)]
        beta_space = [5.0]
        noise_std_space = [0.1]
        cpr = 0.1
    elif hyper_mode == 'X12':
        lr_space = [0.0001]
        train_mode_space = [(32,100,-1)]
        beta_space = [10.0]
        noise_std_space = [0.1]
        cpr = 0.5
    elif hyper_mode == 'X13':
        lr_space = [0.0001]
        train_mode_space = [(32,100,-1)]
        beta_space = [10.0]
        noise_std_space = [0.1]
        cpr = 0.1
    elif hyper_mode == 'X14':
        lr_space = [0.001]
        train_mode_space = [(32,100,-1)]
        beta_space = [0.01]
        noise_std_space = [0.1]
        cpr = 0.5
    elif hyper_mode == 'X15':
        lr_space = [0.001]
        train_mode_space = [(32,100,-1)]
        beta_space = [0.01]
        noise_std_space = [0.1]
        cpr = 0.1
    elif hyper_mode == 'X16':
        lr_space = [0.001]
        train_mode_space = [(32,100,-1)]
        beta_space = [0.1]
        noise_std_space = [0.1]
        cpr = 0.5
    elif hyper_mode == 'X17':
        lr_space = [0.001]
        train_mode_space = [(32,100,-1)]
        beta_space = [0.1]
        noise_std_space = [0.1]
        cpr = 0.1
    elif hyper_mode == 'X18':
        lr_space = [0.001]
        train_mode_space = [(32,100,-1)]
        beta_space = [0.05]
        noise_std_space = [0.1]
        cpr = 0.5
    elif hyper_mode == 'X19':
        lr_space = [0.001]
        train_mode_space = [(32,100,-1)]
        beta_space = [0.05]
        noise_std_space = [0.1]
        cpr = 0.1
    elif hyper_mode == 'X20':
        lr_space = [0.001]
        train_mode_space = [(32,100,-1)]
        beta_space = [0.5]
        noise_std_space = [0.1]
        cpr = 0.5
    elif hyper_mode == 'X21':
        lr_space = [0.001]
        train_mode_space = [(32,100,-1)]
        beta_space = [0.5]
        noise_std_space = [0.1]
        cpr = 0.1
    elif hyper_mode == 'X22':
        lr_space = [0.001]
        train_mode_space = [(32,100,-1)]
        beta_space = [1.0]
        noise_std_space = [0.1]
        cpr = 0.5
    elif hyper_mode == 'X23':
        lr_space = [0.001]
        train_mode_space = [(32,100,-1)]
        beta_space = [1.0]
        noise_std_space = [0.1]
        cpr = 0.1
    elif hyper_mode == 'X24':
        lr_space = [0.001]
        train_mode_space = [(32,100,-1)]
        beta_space = [5.0]
        noise_std_space = [0.1]
        cpr = 0.5
    elif hyper_mode == 'X25':
        lr_space = [0.001]
        train_mode_space = [(32,100,-1)]
        beta_space = [5.0]
        noise_std_space = [0.1]
        cpr = 0.1
    elif hyper_mode == 'X26':
        lr_space = [0.001]
        train_mode_space = [(32,100,-1)]
        beta_space = [10.0]
        noise_std_space = [0.1]
        cpr = 0.5
    elif hyper_mode == 'X27':
        lr_space = [0.001]
        train_mode_space = [(32,100,-1)]
        beta_space = [10.0]
        noise_std_space = [0.1]
        cpr = 0.1

    else:
        sys.exit('Wrong Hyper mode!! Inpuy hyper mode!!')
    commands = []
    algo_group_ = algo_group.split('_')
    if algo_group_[0] in ['ApproxNeuraLCB' ,'ApproxNeuraLCBV2']:
        for lr in lr_space:
            for batch_size,num_steps,buffer_s in train_mode_space:
                for beta in beta_space:
                    for noise_std in noise_std_space:
                        if test == 1:
                            commands.append(
                                f'python realworld_main.py '
                                f'--num_train_sepsis_pat_win 5 '
                                f'--num_test_pat_septic_win 1 '
                                f'--data_type {data_type} '
                                f'--algo_group {algo_group} '
                                f'--num_sim {num_sim} '
                                f'--batch_size {batch_size} '
                                f'--num_steps {num_steps} '
                                f'--buffer_s {buffer_s} '
                                f'--beta {beta} '
                                f'--lr {lr} '
                                f'--policy {policy} '
                                f'--noise_std {noise_std} '
                                f'--group {group} '
                                f'--test_freq {test_freq}'
                            )
                        else:
                            commands.append(
                                f'python realworld_main.py '
                                f'--data_type {data_type} '
                                f'--algo_group {algo_group} '
                                f'--num_sim {num_sim} '
                                f'--batch_size {batch_size} '
                                f'--num_steps {num_steps} '
                                f'--buffer_s {buffer_s} '
                                f'--beta {beta} '
                                f'--lr {lr} '
                                f'--policy {policy} '
                                f'--noise_std {noise_std} '
                                f'--group {group} '
                                f'--test_freq {test_freq}'
                            )
    if algo_group_[0] in ['ApproxNeuralLinLCBV2', 'ApproxNeuralLinLCBJointModel']:
        for lr in lr_space:
            for batch_size,num_steps,buffer_s in train_mode_space:
                for beta in beta_space:
                    for noise_std in noise_std_space:
                        if test == 1:
                            commands.append(
                                f'python realworld_main.py '
                                f'--num_train_sepsis_pat_win 5 '
                                f'--num_test_pat_septic_win 1 '
                                f'--data_type {data_type} '
                                f'--algo_group {algo_group} '
                                f'--num_sim {num_sim} '
                                f'--batch_size {batch_size} '
                                f'--num_steps {num_steps} '
                                f'--buffer_s {buffer_s} '
                                f'--beta {beta} '
                                f'--lr {lr} '
                                f'--policy {policy} '
                                f'--noise_std {noise_std} '
                                f'--group {group} '
                                f'--cpr {cpr} '
                                f'--test_freq {test_freq}'

                            )
                        else:
                            commands.append(
                                f'python realworld_main.py '
                                f'--data_type {data_type} '
                                f'--algo_group {algo_group} '
                                f'--num_sim {num_sim} '
                                f'--batch_size {batch_size} '
                                f'--num_steps {num_steps} '
                                f'--buffer_s {buffer_s} '
                                f'--beta {beta} '
                                f'--lr {lr} '
                                f'--policy {policy} '
                                f'--noise_std {noise_std} '
                                f'--group {group} '
                                f'--cpr {cpr} '
                                f'--test_freq {test_freq}'
                            )
    elif algo_group == 'NeuralGreedyV2_cp' or 'NeuralGreedyV2': # 'NeuralGreedyV2_cp' does not have beta
        for lr in lr_space:
            for batch_size,num_steps,buffer_s in train_mode_space:
                for noise_std in noise_std_space:
                    if test:
                        commands.append(
                            f'python realworld_main.py '
                            f'--num_train_sepsis_pat_win 5 '
                            f'--num_test_pat_septic_win 1 '
                            f'--data_type {data_type} '
                            f'--algo_group {algo_group} '
                            f'--num_sim {num_sim} '
                            f'--batch_size {batch_size} '
                            f'--num_steps {num_steps} '
                            f'--buffer_s {buffer_s} '
                            f'--lr {lr} '
                            f'--policy {policy} '
                            f'--noise_std {noise_std} '
                            f'--group {group} '
                            f'--test_freq {test_freq}'
                        )
                    else:
                        commands.append(
                            f'python realworld_main.py '
                            f'--data_type {data_type} '
                            f'--algo_group {algo_group} '
                            f'--num_sim {num_sim} '
                            f'--batch_size {batch_size} '
                            f'--num_steps {num_steps} '
                            f'--buffer_s {buffer_s} '
                            f'--lr {lr} '
                            f'--policy {policy} '
                            f'--noise_std {noise_std} '
                            f'--group {group} '
                            f'--test_freq {test_freq}'
                        )
    else:
        raise NotImplementedError

    return commands

def run_exps():
    commands = []
    for data_type in args.data_types:
        for algo_group in args.algo_groups:
            for policy in args.policies:
                commands += create_commands(data_type, algo_group, args.num_sim, policy, args.mode,args.G, args.test, args.test_freq)
    random.shuffle(commands)
    multi_gpu_launcher(commands, args.gpus, args.models_per_gpu)


if __name__ == '__main__':
    eval(args.task)()
