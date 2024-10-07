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


from core.contextual_bandit import contextual_bandit_runner, contextual_bandit_runner_v2

from algorithms.neural_offline_bandit_cp import ApproxNeuraLCB_cp 
from algorithms.comparison_bandit_cp import ExactNeuraLCBV2_cp, NeuralGreedyV2_cp, NeuraLCB_cp
from algorithms.neural_lin_lcb_cp import ApproxNeuralLinLCBV2_cp, ApproxNeuralLinLCBJointModel_cp
 
from algorithms.neural_offline_bandit import ExactNeuraLCBV2, NeuralGreedyV2, ApproxNeuraLCBV2
from algorithms.lin_lcb import LinLCB 
from algorithms.kern_lcb import KernLCB 
from algorithms.uniform_sampling import UniformSampling
from algorithms.neural_lin_lcb import ExactNeuralLinLCBV2, ExactNeuralLinGreedyV2, ApproxNeuralLinLCBV2, ApproxNeuralLinGreedyV2, \
    ApproxNeuralLinLCBJointModel, NeuralLinGreedyJointModel

from algorithms.neural_lin_lcb_cp import ApproxNeuralLinLCBV2_cp,ExactNeuralLinLCBV2_cp
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
import PI_Sepsysolcp as EnbPI
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
# flags.DEFINE_string('algo_group', 'ExactNeuraLCBV2_cp', 'conformal prediction/neural')
# flags.DEFINE_string('algo_group', 'NeuralGreedyV2_cp', 'conformal prediction/neural')
flags.DEFINE_boolean('is_window', True, 'to use the window sized data or not?') 

flags.DEFINE_integer('num_train_sepsis_pat_win', 10 , 'Number of septic windows for training.') 
flags.DEFINE_integer('num_test_pat_septic_win', 1, 'Number of septic windows for testing.') 
flags.DEFINE_integer('win_size', 8, 'Window size used for training and testing.')

 

# flags.DEFINE_string('data_type', 'mushroom', 'Dataset to sample from')

# let's test on mnistm cuz this is the only dataset available/opened

    # dataclasses = {'mushroom':MushroomData, 'jester':JesterData, 'statlog':StatlogData, 'covertype':CoverTypeData, 'stock': StockData,
    #         'adult': AdultData, 'census': CensusData, 'mnist': MnistData
    # }

# flags.DEFINE_string('data_type', 'mnist', 'Dataset to sample from')
# flags.DEFINE_string('data_type', 'covertype', 'Dataset to sample from')
flags.DEFINE_string('data_type', 'sepsis', 'Dataset to sample from')

flags.DEFINE_string('policy', 'eps-greedy', 'Offline policy, eps-greedy/subset')
# flags.DEFINE_string('policy', 'online', 'Offline policy, eps-greedy/subset')
flags.DEFINE_float('eps', 0.1, 'Probability of selecting a random action in eps-greedy')
flags.DEFINE_float('subset_r', 0.5, 'The ratio of the action spaces to be selected in offline data')


# num_train_sepsis_pat_win = 20
# num_test_pat_septic_win = 5
# num_train_sepsis_pat_win = 1000
# num_test_pat_septic_win = 250
# # win_size= 8 
# print(f'num_train_sepsis_pat_win === {num_train_sepsis_pat_win}')
# print(f'num_test_pat_septic_win === {num_test_pat_septic_win}')






# is_window = True
# # this might only corresponding to a few hundreds patients
# if is_window:

#     flags.DEFINE_integer('num_contexts', num_train_sepsis_pat_win*win_size*2, 'Number of contexts for training.') 
#     flags.DEFINE_integer('num_test_contexts', num_test_pat_septic_win*win_size*13, 'Number of contexts for test.') 
# else:

#     flags.DEFINE_integer('num_contexts', 500, 'Number of contexts for training.') 
#     flags.DEFINE_integer('num_test_contexts', 100, 'Number of contexts for test.') 

flags.DEFINE_boolean('verbose', True, 'verbose') 
flags.DEFINE_boolean('debug', True, 'debug') 
flags.DEFINE_boolean('normalize', False, 'normalize the regret') 
flags.DEFINE_integer('update_freq', 1, 'Update frequency')
flags.DEFINE_integer('freq_summary', 10, 'Summary frequency')

flags.DEFINE_integer('test_freq', 10, 'Test frequency')
# flags.DEFINE_string('algo_group', 'approx-neural', 'baseline/neural')






# flags.DEFINE_integer('num_sim', 10, 'Number of simulations')
flags.DEFINE_integer('num_sim', 1, 'Number of simulations')

flags.DEFINE_float('noise_std', 0.01, 'Noise std')

flags.DEFINE_integer('chunk_size', 500, 'Chunk size')
# flags.DEFINE_integer('chunk_size', 5, 'Chunk size')
flags.DEFINE_integer('batch_size', 32, 'Batch size')
# flags.DEFINE_integer('batch_size', 4, 'Batch size')
flags.DEFINE_integer('num_steps', 100, 'Number of steps to train NN.') 
# flags.DEFINE_integer('num_steps', 10, 'Number of steps to train NN.') 

 
flags.DEFINE_integer('buffer_s', -1, 'Size in the train data buffer.')
flags.DEFINE_bool('data_rand', True, 'Where randomly sample a data batch or  use the latest samples in the buffer' )

flags.DEFINE_float('rbf_sigma', 1, 'RBF sigma for KernLCB') # [0.1, 1, 10]

# NeuraLCB 
flags.DEFINE_float('beta', 0.1, 'confidence paramter') # [0.01, 0.05, 0.1, 0.5, 1, 5, 10] 
flags.DEFINE_float('lr', 1e-3, 'learning rate') 
flags.DEFINE_float('lambd0', 0.1, 'minimum eigenvalue') 
flags.DEFINE_float('lambd', 1e-4, 'regularization parameter')

#================================================================
# Network parameters
#================================================================
def main(unused_argv): 

    # is_window = True
    # this might only corresponding to a few hundreds patients
    if FLAGS.is_window:

        # flags.DEFINE_integer('num_contexts', num_train_sepsis_pat_win*win_size*2, 'Number of contexts for training.') 
        # flags.DEFINE_integer('num_test_contexts', num_test_pat_septic_win*win_size*13, 'Number of contexts for test.') 
        num_contexts = FLAGS.num_train_sepsis_pat_win * FLAGS.win_size * 2
        num_test_contexts = FLAGS.num_test_pat_septic_win * FLAGS.win_size * 13
    else:

        # flags.DEFINE_integer('num_contexts', 500, 'Number of contexts for training.') 
        # flags.DEFINE_integer('num_test_contexts', 100, 'Number of contexts for test.') 
        num_contexts = 500
        num_test_contexts = 300
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

# different datasets
    dataclasses = {'mushroom':MushroomData, 'jester':JesterData, 'statlog':StatlogData, 'covertype':CoverTypeData, 'stock': StockData,
            'adult': AdultData, 'census': CensusData, 'mnist': MnistData, 'sepsis': SepsisData
    }
    
    if FLAGS.data_type in dataclasses:
        # so actually this is returning a class not a string
        DataClass = dataclasses[FLAGS.data_type]
        if FLAGS.data_type == 'sepsis':
        # class SepsisData(object):
        #     def __init__(self,  
        #                 is_window,
        #                 num_train_sepsis_pat_win,
        #                 num_test_pat_septic_win,
        #                 num_contexts, 
        #                 num_test_contexts,         
        #                 num_actions=2, 
        #                 noise_std=0.01,
        #                 pi='eps-greedy', 
        #                 eps=0.1, 
        #                 subset_r=0.5
        #                 ):
            data = DataClass(                
                        is_window = FLAGS.is_window,
                        num_train_sepsis_pat_win= FLAGS.num_train_sepsis_pat_win,
                        num_test_pat_septic_win= FLAGS.num_test_pat_septic_win, 
                        # num_contexts=FLAGS.num_contexts, 
                        num_contexts=num_contexts, 
                        # num_test_contexts=FLAGS.num_test_contexts,
                        num_test_contexts=num_test_contexts,
                        num_actions=2,
                        noise_std = FLAGS.noise_std,
                        pi = FLAGS.policy, 
                        eps = FLAGS.eps, 
                        subset_r = FLAGS.subset_r) 
        else:

            data = DataClass(
                   # num_contexts=FLAGS.num_contexts, 
                     num_contexts=num_contexts, 
                        # num_test_contexts=FLAGS.num_test_contexts,
                        num_test_contexts= num_test_contexts,
                        pi = FLAGS.policy, 
                        eps = FLAGS.eps, 
                        subset_r = FLAGS.subset_r) 
    else:
        raise NotImplementedError

    dataset = data.reset_data()
    context_dim = dataset[0].shape[1] 
    print(f'context_dim: {context_dim}')
  
    context_dim = dataset[0].shape[1] 
    num_actions = data.num_actions 
   

    
    hparams = edict({
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
        'max_test_batch': FLAGS.batch_size
        # 'policy_prefix', policy_prefix
    })

    lin_hparams = edict(
        {
            'context_dim': hparams.context_dim, 
            'num_actions': hparams.num_actions, 
            'lambd0': hparams.lambd0, 
            'beta': hparams.beta, 
            'rbf_sigma': FLAGS.rbf_sigma, # 0.1, 1, 10
            'max_num_sample': 1000 
        }
    )



    data_prefix = '{}_d={}_a={}_pi={}_std={}_testfreq={}'.format(FLAGS.data_type, \
            context_dim, num_actions, policy_prefix, data.noise_std,FLAGS.test_freq)

    # res_dir = os.path.join('results', data_prefix) 
    sim = 0

    res_dir = os.path.join(f'../neuralcb_results/sim{sim}/trainwins_{FLAGS.num_train_sepsis_pat_win}_testwins_{FLAGS.num_test_pat_septic_win}/', data_prefix) 


    flags.DEFINE_string('res_dir', res_dir, 'final result path for each simulation')
    print(f'final result path === {FLAGS.res_dir}')

    if not os.path.exists(res_dir):
        os.makedirs(res_dir)

    

    #================================================================
    # Algorithms 
    #================================================================
    original_stdout = sys.stdout
    print(f'$$$$$########## $$$$$########## algorithm: FLAGS.algo_group $$$$$##########$$$$$##########')
    with open(res_dir+f'/trainwin_{FLAGS.num_train_sepsis_pat_win}test_win_{FLAGS.num_test_pat_septic_win}_{FLAGS.algo_group}_log.txt', 'w') as f:
        sys.stdout = f 
    # if res_dir:

        # if FLAGS.algo_group == 'approx-neural':
        # if FLAGS.algo_group == 'approx-neural_cp':
        #     algos = [
        #         # class UniformSampling(BanditAlgorithm)
        #             # UniformSampling(lin_hparams),
        #             # NeuralGreedyV2(hparams, update_freq = FLAGS.update_freq), 
        #             # class ApproxNeuraLCBV2(BanditAlgorithm)
        #             ApproxNeuraLCB_cp(hparams, res_dir = FLAGS.res_dir, update_freq = FLAGS.update_freq)
        #             # ApproxNeuraLCBV2(hparams, update_freq = FLAGS.update_freq)
        #         ]
        #     algo_prefix = 'approx-neural-gridsearch_epochs={}_m={}_layern={}_buffer={}_bs={}_lr={}_beta={}_lambda={}_lambda0={}'.format(
        #         hparams.num_steps, min(hparams.layer_sizes), hparams.layer_n, hparams.buffer_s, hparams.batch_size, hparams.lr, \
        #         hparams.beta, hparams.lambd, hparams.lambd0
        #     )
 

        if FLAGS.algo_group == 'kern': # for tuning KernLCB
            algos = [
                UniformSampling(lin_hparams),
                KernLCB(lin_hparams), 
            ]

            algo_prefix = 'kern-gridsearch_beta={}_rbf-sigma={}_maxnum={}'.format(
                hparams.beta, lin_hparams.rbf_sigma, lin_hparams.max_num_sample
            )

        if FLAGS.algo_group == 'neurallinlcb': # Tune NeuralLinLCB seperately  
            algos = [
                UniformSampling(lin_hparams),
                ApproxNeuralLinLCBJointModel(hparams)
            ]

            algo_prefix = 'neurallinlcb-gridsearch_m={}_layern={}_beta={}_lambda0={}'.format(
                min(hparams.layer_sizes), hparams.layer_n, hparams.beta, hparams.lambd0
            )


        # Create a dictionary to map algo_group names to their respective classes
        # ALGO_MAP = {
        #     'ExactNeuraLCBV2_cp': ExactNeuraLCBV2_cp,
        #     'NeuralGreedyV2_cp': NeuralGreedyV2_cp,
        #     'ApproxNeuraLCB_cp': ApproxNeuraLCB_cp,
        #     'NeuraLCB_cp': NeuraLCB_cp,
        #     'ApproxNeuralLinLCBV2_cp': ApproxNeuralLinLCBV2_cp,
        #     'ExactNeuralLinLCBV2_cp': ExactNeuralLinLCBV2_cp,
        #     'ApproxNeuralLinLCBJointModel_cp': ApproxNeuralLinLCBJointModel_cp
        # }


        ALGO_MAP = {
            'NeuralGreedyV2_cp': NeuralGreedyV2_cp,
            'ApproxNeuralLinLCBV2_cp': ApproxNeuralLinLCBV2_cp,
            'ApproxNeuralLinLCBJointModel_cp': ApproxNeuralLinLCBJointModel_cp,
            'ApproxNeuraLCB_cp': ApproxNeuraLCB_cp, # finished already
            'ExactNeuraLCBV2_cp': ExactNeuraLCBV2_cp, # run if we have time
	        'ExactNeuralLinLCBV2_cp': ExactNeuralLinLCBV2_cp  # run if we have time
        }
        if FLAGS.algo_group not in ALGO_MAP:
            raise ValueError(f"Unknown algo_group: {FLAGS.algo_group}")
        else:
            print(f'@@@@@@@@@~~~~~~~~~~~~~~~ Algorithm Testing ==== {FLAGS.algo_group}~~~~~~~~~~~~~~~@@@@@@@@@')

        # Get the algorithm class
        algo_class = ALGO_MAP[FLAGS.algo_group]

        # Create the algorithm instance
        algos = [
            algo_class(hparams, res_dir=FLAGS.res_dir, update_freq=FLAGS.update_freq)
        ]

        # Create the prefix string using f-string
        algo_prefix = (
            f"{FLAGS.algo_group}-gridsearch_epochs={hparams.num_steps}_m={min(hparams.layer_sizes)}"
            f"_layern={hparams.layer_n}_buffer={hparams.buffer_s}_bs={hparams.batch_size}"
            f"_lr={hparams.lr}_beta={hparams.beta}_lambda={hparams.lambd}_lambda0={hparams.lambd0}"
        )


        
        #==============================
        # Runner 
        #==============================
        # file path for saving the results
        file_name = os.path.join(res_dir, algo_prefix) + '.npz' 
        # file_path = os.path.join(res_dir, algo_prefix)
        
        
        # this is the core function that run all the experiments
        
        start  =  time.time()
        # regrets, errs = contextual_bandit_runner(algos, data, FLAGS.num_sim, FLAGS.update_freq, FLAGS.test_freq, FLAGS.verbose, FLAGS.debug, FLAGS.normalize, file_name, res_dir)
        regrets, errs = contextual_bandit_runner_v2(algos, data, \
            FLAGS.num_sim, FLAGS.test_freq, FLAGS.verbose, FLAGS.debug, FLAGS.normalize, res_dir,algo_prefix,file_name,sim)

        np.savez(file_name, regrets=regrets, errs=errs)

        print(f'total time for contextual bandit runner: {time.time()-start} seconds')




    sys.stdout = original_stdout


# thissetup is only executed only if the script is run directly from the command line, not when imported as a module in another python project scrpit.
# app() ensures that all the command-line arguments are parsed
if __name__ == '__main__': 
    app.run(main)

        