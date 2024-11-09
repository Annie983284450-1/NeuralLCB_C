"""Mini main for testing algorithms. """ 

import numpy as np 
import jax  
import jax.numpy as jnp 
from easydict import EasyDict as edict
import os 
import importlib

import core.contextual_bandit  # Import the entire module

# Reload the module
importlib.reload(core.contextual_bandit)


from core.contextual_bandit import contextual_bandit_runner, contextual_bandit_runner_v2,contextual_bandit_runner_v3

from algorithms.neural_offline_bandit_cp import ApproxNeuraLCB_cp 
from algorithms.comparison_bandit_cp import ExactNeuraLCBV2_cp, NeuralGreedyV2_cp, NeuraLCB_cp
from algorithms.neural_lin_lcb_cp import ApproxNeuralLinLCBV2_cp, ApproxNeuralLinLCBJointModel_cp,ExactNeuralLinLCBV2_cp
 
from algorithms.neural_offline_bandit import ExactNeuraLCBV2, NeuralGreedyV2, ApproxNeuraLCBV2
from algorithms.lin_lcb import LinLCB 
from algorithms.kern_lcb import KernLCB 
from algorithms.uniform_sampling import UniformSampling
from algorithms.neural_lin_lcb import ExactNeuralLinLCBV2, ExactNeuralLinGreedyV2, ApproxNeuralLinLCBV2, ApproxNeuralLinGreedyV2, \
    ApproxNeuralLinLCBJointModel, NeuralLinGreedyJointModel

# from algorithms.neural_lin_lcb_cp import ApproxNeuralLinLCBV2_cp,ExactNeuralLinLCBV2_cp
# from algorithms.neural_lin_lcb import ApproxNeuralLinLCBJointModel, ApproxNeuralLinLCBV2, ApproxNeuralLinGreedyV2
# data class is defined in this script!!
from data.realworld_data import *
from data.sepsisdataclass import *

from absl import flags, app

import time
import math
import random
import pickle
# import utils_ECAD_journal as utils_ECAD
from scipy.stats import skew
import seaborn as sns
# import PI_class_EnbPI_journal as EnbPI
# import PI_Sepsysolcp as EnbPI
import matplotlib
# matplotlib.use('TkAgg',force=True)
# matplotlib.use('Agg')
import os
if os.environ.get('DISPLAY','') == '':
    print('No display found. Using non-interactive Agg backend')
    matplotlib.use('Agg')
else:
    matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
print("Switched to:",matplotlib.get_backend())
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from pyod.models.pca import PCA
from pyod.models.ocsvm import OCSVM
from pyod.models.iforest import IForest
from pyod.models.hbos import HBOS
from pyod.models.knn import KNN   # kNN detector
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVR
from sklearn.model_selection import RandomizedSearchCV
from xgboost import XGBRegressor

from sklearn import neighbors
from sklearn.neural_network import MLPClassifier
# from sklearn import svm
import utils_Sepsysolcp as util
from matplotlib.lines import Line2D  # For legend handles
# import calendar
import warnings
import matplotlib.pyplot as plt
from sklearn.linear_model import RidgeCV, LassoCV
from sklearn.ensemble import RandomForestRegressor

# FLAGS will store values of all command-line flags defined and passed by the users
FLAGS = flags.FLAGS 

'''
flags.DEFINE_<type>(name, default, help_text) 
<type>: data type of the flag
name: name of the flag
default: the default value of the flag if not specified by the user
help_text: string describes the flag

'''

# flags.DEFINE_string('algo_group', 'approx-neural_cp', 'conformal prediction/neural')
flags.DEFINE_string('algo_group', 'ApproxNeuraLCB_cp', 'conformal prediction/neural')
flags.DEFINE_string('data_type', 'sepsis1', 'Dataset to sample from')
flags.DEFINE_string('policy', 'eps-greedy', 'Offline policy, eps-greedy/subset')
flags.DEFINE_string('group', 'septic', 'Testing data group')


# used for 8 hrs window

flags.DEFINE_integer('num_train_sepsis_pat_win', 2000, 'Number of septic windows for training.') 
flags.DEFINE_integer('num_test_pat_septic_win', 250, 'Number of septic windows for testing.') 

# used for streaming patient one by one 
flags.DEFINE_integer('num_train_pat_septic', 5, 'Number of septic patients for training.') 
flags.DEFINE_integer('num_test_pat_septic', 2, 'Number of septic patients for training.') 




flags.DEFINE_integer('win_size', 8, 'Window size used for training and testing.')
flags.DEFINE_integer('B', 10, 'number of bootstraps')
flags.DEFINE_integer('update_freq', 1, 'Update frequency')
flags.DEFINE_integer('freq_summary', 10, 'Summary frequency')
flags.DEFINE_integer('test_freq', 100, 'Test frequency')
# flags.DEFINE_integer('num_sim', 10, 'Number of simulations')
flags.DEFINE_integer('num_sim', 1, 'Number of simulations')
flags.DEFINE_integer('chunk_size', 500, 'Chunk size')
# flags.DEFINE_integer('chunk_size', 5, 'Chunk size')
flags.DEFINE_integer('batch_size', 32, 'Batch size')
# flags.DEFINE_integer('batch_size', 4, 'Batch size')
flags.DEFINE_integer('num_steps', 100, 'Number of steps to train NN.') 
# flags.DEFINE_integer('num_steps', 10, 'Number of steps to train NN.') 
flags.DEFINE_integer('buffer_s', -1, 'Size in the train data buffer.')

flags.DEFINE_float('eps', 0.1, 'Probability of selecting a random action in eps-greedy')
flags.DEFINE_float('subset_r', 0.5, 'The ratio of the action spaces to be selected in offline data')
flags.DEFINE_float('noise_std', 0.01, 'Noise std')
flags.DEFINE_float('rbf_sigma', 1, 'RBF sigma for KernLCB') # [0.1, 1, 10]
# NeuraLCB 
flags.DEFINE_float('beta', 0.1, 'confidence paramter') # [0.01, 0.05, 0.1, 0.5, 1, 5, 10] 
flags.DEFINE_float('lr', 1e-3, 'learning rate') 
flags.DEFINE_float('lambd0', 0.1, 'minimum eigenvalue') 
flags.DEFINE_float('lambd', 1e-4, 'regularization parameter')


flags.DEFINE_bool('is_window', True, 'to use the window sized data or not?') 
flags.DEFINE_bool('verbose', True, 'verbose') 
flags.DEFINE_bool('debug', True, 'debug') 
flags.DEFINE_bool('normalize', False, 'normalize the regret')
flags.DEFINE_bool('data_rand', True, 'Where randomly sample a data batch or  use the latest samples in the buffer' )


#================================================================
# Network parameters
#================================================================
def main(unused_argv): 

    # is_window = True
    # this might only corresponding to a few hundreds patients
    # if FLAGS.is_window:
    if FLAGS.data_type == 'sepsis':

        num_contexts = FLAGS.num_train_sepsis_pat_win * FLAGS.win_size * 2
        num_test_contexts = FLAGS.num_test_pat_septic_win * FLAGS.win_size * 13
 
    #=================
    # Data 
    #=================
    if FLAGS.policy == 'eps-greedy':
        policy_prefix = '{}{}'.format(FLAGS.policy, FLAGS.eps)
    elif FLAGS.policy == 'subset':
        policy_prefix = '{}{}'.format(FLAGS.policy, FLAGS.subset_ratio)
    elif FLAGS.policy == 'online':
        policy_prefix = '{}{}'.format(FLAGS.policy, FLAGS.eps) 
    else:


        raise NotImplementedError('{} not implemented'.format(FLAGS.policy))


    dataclasses = {'sepsis': SepsisData, 'sepsis1': SepsisData1}
    
    if FLAGS.data_type in dataclasses:
        # so actually this is returning a class not a string
        DataClass = dataclasses[FLAGS.data_type]
        if FLAGS.data_type == 'sepsis':
            data = DataClass(                
                        is_window = FLAGS.is_window,
                        num_train_sepsis_pat_win= FLAGS.num_train_sepsis_pat_win,
                        num_test_pat_septic_win= FLAGS.num_test_pat_septic_win, 
                        num_contexts=num_contexts, 
                        num_test_contexts=num_test_contexts,
                        num_actions=2,
                        noise_std = FLAGS.noise_std,
                        pi = FLAGS.policy, 
                        eps = FLAGS.eps, 
                        subset_r = FLAGS.subset_r,
                        group = FLAGS.group) 
        elif FLAGS.data_type == 'sepsis1': # process the pat one by one
            data = DataClass(
                num_actions=2, 
                noise_std=FLAGS.noise_std,
                pi=FLAGS.policy, 
                eps=FLAGS.eps, 
                subset_r=FLAGS.subset_r 
            )
    else:
        raise NotImplementedError
    
    if FLAGS.data_type != 'sepsis1':
        dataset = data.reset_data()
        context_dim = dataset[0].shape[1] 
    else:
        context_dim = 13
    num_actions = data.num_actions 
   

    
    hparams = edict({
        # (m,m)
        'layer_sizes': [100,100], 
        # 'layer_sizes': [5,5], 
        's_init': 1, 
        'activation': jax.nn.relu, 
        'layer_n': True,
        'seed': 0,
        'context_dim': context_dim, 
        'num_actions': num_actions, 
        'beta': FLAGS.beta, # [0.01, 0.05, 0.1, 0.5, 1, 5, 10]
        'lambd': FLAGS.lambd, # regularization param: [0.1m, m, 10 m  ]
        'lr': FLAGS.lr, 
        'lambd0': FLAGS.lambd0, # shoud be lambd/m in theory but we fix this at 0.1 for simplicity and mainly focus on tuning beta 
        'verbose': False, 
        'batch_size': FLAGS.batch_size,
        'freq_summary': FLAGS.freq_summary, 
        'chunk_size': FLAGS.chunk_size, 
        'num_steps': FLAGS.num_steps, 
        'buffer_s': FLAGS.buffer_s, 
        'data_rand': FLAGS.data_rand,
        'debug_mode': 'full', # simple/full
        'num_train_sepsis_pat_win': FLAGS.num_train_sepsis_pat_win,
        'num_test_pat_septic_win': FLAGS.num_test_pat_septic_win,
        'data_type':FLAGS.data_type,
        'max_test_batch': FLAGS.batch_size,
        'B': FLAGS.B
    })


    data_prefix = '{}_d={}_a={}_pi={}_std={}_testfreq={}'\
        .format(FLAGS.data_type, context_dim, num_actions, policy_prefix, data.noise_std,FLAGS.test_freq)

    # res_dir = os.path.join('results', data_prefix) 
    sim = 0
    res_dir = os.path.join(
        f'/storage/home/hcoda1/6/azhou60/scratch/neuralcb_results/sim{sim}/'
        f'trainSepticWins_{FLAGS.num_train_sepsis_pat_win}_'
        f'testSepticWins_{FLAGS.num_test_pat_septic_win}/',
        data_prefix
    )


    flags.DEFINE_string('res_dir', res_dir, 'final result path for each simulation')
    print(f'final result path === {FLAGS.res_dir}')

    if not os.path.exists(res_dir):
        os.makedirs(res_dir, exist_ok=True)

    

    #================================================================
    # Algorithms 
    #================================================================

    ALGO_MAP_cp = {
        'NeuralGreedyV2_cp': NeuralGreedyV2_cp,
        'ApproxNeuralLinLCBV2_cp': ApproxNeuralLinLCBV2_cp,
        'ApproxNeuralLinLCBJointModel_cp': ApproxNeuralLinLCBJointModel_cp,
        'ApproxNeuraLCB_cp': ApproxNeuraLCB_cp, # finished already
        'ExactNeuraLCBV2_cp': ExactNeuraLCBV2_cp, # run if we have time
        'ExactNeuralLinLCBV2_cp': ExactNeuralLinLCBV2_cp  # run if we have time   
    }

    ALGO_MAP = {
        'NeuralGreedyV2': NeuralGreedyV2,
        'ApproxNeuralLinLCBV2': ApproxNeuralLinLCBV2,
        'ApproxNeuralLinLCBJointModel': ApproxNeuralLinLCBJointModel,
        'ExactNeuraLCBV2': ExactNeuraLCBV2, # run if we have time
        'ExactNeuralLinLCBV2': ExactNeuralLinLCBV2,  # run if we have time
        'ApproxNeuraLCBV2': ApproxNeuraLCBV2
    }

    if FLAGS.algo_group in ALGO_MAP_cp:
        # raise ValueError(f"Unknown algo_group: {FLAGS.algo_group}")
        print(f'@@@@@@@@@~~~~~~~~~~~~~~~ Algorithm Testing ==== {FLAGS.algo_group}~~~~~~~~~~~~~~~@@@@@@@@@')
    
        # Get the algorithm class
        algo_class = ALGO_MAP_cp[FLAGS.algo_group]

        # Create the algorithm instance
        algos = [
            algo_class(hparams, res_dir=FLAGS.res_dir, B = FLAGS.B, update_freq=FLAGS.update_freq)
        ]
        # Create the prefix string using f-string
        algo_prefix = (
            f"{FLAGS.algo_group}-gridsearch_epochs={hparams.num_steps}_m={min(hparams.layer_sizes)}"
            f"_layern={hparams.layer_n}_buffer={hparams.buffer_s}_bs={hparams.batch_size}"
            f"_lr={hparams.lr}_beta={hparams.beta}_lambda={hparams.lambd}_lambda0={hparams.lambd0}"
            f"_B={hparams.B}"
            f"_G={FLAGS.group}"
        )
        # nohup_output = res_dir+f'/trainwin_{FLAGS.num_train_sepsis_pat_win}test_win_{FLAGS.num_test_pat_septic_win}_{FLAGS.algo_group}_B={FLAGS.B}_log.txt'
        if not os.path.exists(os.path.join(res_dir, algo_prefix)):
            os.makedirs(os.path.join(res_dir, algo_prefix), exist_ok=True)
        nohup_output = os.path.join(res_dir, algo_prefix) + '/'+algo_prefix+'.log' 
    elif FLAGS.algo_group in ALGO_MAP:
        print(f'@@@@@@@@@~~~~~~~~~~~~~~~ Algorithm Testing ==== {FLAGS.algo_group}~~~~~~~~~~~~~~~@@@@@@@@@')
        # Get the algorithm class
        algo_class = ALGO_MAP[FLAGS.algo_group]

        # Create the algorithm instance
        algos = [
            algo_class(hparams, update_freq=FLAGS.update_freq)
        ]
        # Create the prefix string using f-string
        algo_prefix = (
            f"{FLAGS.algo_group}-gridsearch_epochs={hparams.num_steps}_m={min(hparams.layer_sizes)}"
            f"_layern={hparams.layer_n}_buffer={hparams.buffer_s}_bs={hparams.batch_size}"
            f"_lr={hparams.lr}_beta={hparams.beta}_lambda={hparams.lambd}_lambda0={hparams.lambd0}"
            f"_G={FLAGS.group}"
        )
        # nohup_output = res_dir+f'/trainwin_{FLAGS.num_train_sepsis_pat_win}test_win_{FLAGS.num_test_pat_septic_win}_{FLAGS.algo_group}_B={hparams.B}log.txt'
        if not os.path.exists(os.path.join(res_dir, algo_prefix)):
            os.makedirs(os.path.join(res_dir, algo_prefix), exist_ok=True)
        nohup_output = os.path.join(res_dir, algo_prefix) + '/'+ algo_prefix+'.log' 
    else:
        raise ValueError(f"Unknown algo_group: {FLAGS.algo_group}")
    
    file_name = os.path.join(res_dir, algo_prefix) + '/' + algo_prefix+'.npz' 
    

    original_stdout = sys.stdout
    print(f'$$$$$########## $$$$$########## algorithm: {FLAGS.algo_group}$$$$$##########$$$$$##########')
    start_runner= time.time()
    with open(nohup_output, 'w') as f:
        sys.stdout = f 
    # if res_dir:
        
        #==============================
        # Runner 
        #==============================
        start  =  time.time()
        # regrets, errs = contextual_bandit_runner(algos, data, FLAGS.num_sim, FLAGS.update_freq, FLAGS.test_freq, FLAGS.verbose, FLAGS.debug, FLAGS.normalize, file_name, res_dir)
        if FLAGS.data_type == 'sepsis':
            regrets, errs = contextual_bandit_runner_v2(algos, data, \
                FLAGS.num_sim, FLAGS.test_freq, FLAGS.verbose, FLAGS.debug, FLAGS.normalize, res_dir,algo_prefix,file_name,sim, FLAGS.B)
            np.savez(file_name, regrets=regrets, errs=errs)
            print(f'total time for contextual bandit runner: {time.time()-start} seconds')
            
        elif FLAGS.data_type == 'sepsis1': # runner v3 will process only one patient each time

            #~~~~~~~~~~~~~~Testing set~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            ratio = 1
            # test_sepsis = np.load('./data/SepsisData/test_sepsis.npy')
            # test_nosepsis = np.load('./data/SepsisData/test_nosepsis.npy')
            test_sepsis = np.load('./data/SepsisData/test_sepsis_wins.npy')
            test_nosepsis = np.load('./data/SepsisData/test_nosepsis_wins.npy')
            # test_sepsis = test_sepsis[0: FLAGS.num_test_pat_septic]
            # test_nosepsis = test_nosepsis[0:FLAGS.num_test_pat_septic*ratio]
            test_sepsis = test_sepsis[0: FLAGS.num_test_pat_septic_win]
            test_nosepsis = test_nosepsis[0:FLAGS.num_test_pat_septic_win*ratio]
   
            # test_set_psv = np.concatenate((test_sepsis, test_nosepsis), axis=0) 
            # test_set = [filename.replace('.psv', '') for filename in test_set_psv]
            test_set = np.concatenate((test_sepsis, test_nosepsis), axis=0) 

            np.random.seed(12345)  # Set a seed for reproducibility
            np.random.shuffle(test_set)
            # test_sepsis = [filename.replace('.psv', '') for filename in test_sepsis]
            # test_nosepsis = [filename.replace('.psv', '') for filename in test_nosepsis]
            #~~~~~~~~~~~~~~Training set~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~




            # train_sepsis = np.load('./data/SepsisData/train_sepsis.npy')
            # train_nosepsis = np.load('./data/SepsisData/train_nosepsis.npy')
            train_sepsis = np.load('./data/SepsisData/train_sepsis_wins.npy')
            train_nosepsis = np.load('./data/SepsisData/train_nosepsis_wins.npy')
            # train_sepsis = train_sepsis[0:FLAGS.num_train_pat_septic]
            # train_nosepsis = train_nosepsis[0:FLAGS.num_train_pat_septic]
            train_sepsis = train_sepsis[0:FLAGS.num_train_sepsis_pat_win]
            train_nosepsis = train_nosepsis[0:FLAGS.num_train_sepsis_pat_win]
 
            # train_set_psv = np.concatenate((train_sepsis, train_nosepsis), axis=0)
            # train_patients_ids  = [filename.replace('.psv', '') for filename in train_set_psv]
            train_patients_ids =  np.concatenate((train_sepsis, train_nosepsis), axis=0)
            np.random.seed(12345)  # Set a seed for reproducibility
            np.random.shuffle(train_patients_ids)
            # sepsis_full_df = pd.read_csv('./data/SepsisData/fully_imputed.csv')
            sepsis_full_df = pd.read_csv('./data/SepsisData/fully_imputed_8windowed_max48_updated.csv')
         
            regrets, errs = contextual_bandit_runner_v3(algos, data,  sepsis_full_df, train_patients_ids,test_set,\
                FLAGS.num_sim, FLAGS.test_freq, FLAGS.verbose, FLAGS.debug, FLAGS.normalize, res_dir,algo_prefix,file_name,sim, FLAGS.B)
            np.savez(file_name, regrets=regrets, errs=errs)
            print(f'total time for contextual bandit runner: {time.time()-start} seconds')
            
    sys.stdout = original_stdout
    print(f'Total Excution time: {time.time()-start_runner}')



# thissetup is only executed only if the script is run directly from the command line,
# not when imported as a module in another python project scrpit.
# app() ensures that all the command-line arguments are parsed
if __name__ == '__main__': 
    app.run(main)

        