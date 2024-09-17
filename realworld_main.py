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


from core.contextual_bandit import contextual_bandit_runner
# the core code
# from algorithms.neural_offline_bandit import ExactNeuraLCBV2, NeuralGreedyV2, ApproxNeuraLCBV2


# import algorithms.neural_offline_bandit_cp  # Import the entire module

# # Reload the module
# importlib.reload(algorithms.neural_offline_bandit_cp)
import algorithms.neural_offline_bandit # Import the entire module

# Reload the module
importlib.reload(algorithms.neural_offline_bandit)

# from algorithms.neural_offline_bandit import ExactNeuraLCBV2, NeuralGreedyV2, ApproxNeuraLCBV2_cp
from algorithms.neural_offline_bandit import NeuralGreedyV2, ApproxNeuraLCBV2
from algorithms.neural_offline_bandit_cp import ApproxNeuraLCBV2_cp


from algorithms.lin_lcb import LinLCB 
from algorithms.kern_lcb import KernLCB 
from algorithms.uniform_sampling import UniformSampling
from algorithms.neural_lin_lcb import ExactNeuralLinLCBV2, ExactNeuralLinGreedyV2, ApproxNeuralLinLCBV2, ApproxNeuralLinGreedyV2, \
    ApproxNeuralLinLCBJointModel, NeuralLinGreedyJointModel
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

# seems that everybody has a default value

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
# this might only corresponding to a few hundreds patients
flags.DEFINE_integer('num_contexts', 1500, 'Number of contexts for training.') 
flags.DEFINE_integer('num_test_contexts', 1000, 'Number of contexts for test.') 

# flags.DEFINE_integer('num_contexts', 500, 'Number of contexts for training.') 
# flags.DEFINE_integer('num_test_contexts', 100, 'Number of contexts for test.') 

flags.DEFINE_boolean('verbose', True, 'verbose') 
flags.DEFINE_boolean('debug', True, 'debug') 
flags.DEFINE_boolean('normalize', False, 'normalize the regret') 
flags.DEFINE_integer('update_freq', 1, 'Update frequency')
flags.DEFINE_integer('freq_summary', 10, 'Summary frequency')

flags.DEFINE_integer('test_freq', 10, 'Test frequency')
flags.DEFINE_string('algo_group', 'approx-neural', 'baseline/neural')
# flags.DEFINE_integer('num_sim', 10, 'Number of simulations')
flags.DEFINE_integer('num_sim', 1, 'Number of simulations')

flags.DEFINE_float('noise_std', 0.1, 'Noise std')

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
        data = DataClass(num_contexts=FLAGS.num_contexts, 
                    num_test_contexts=FLAGS.num_test_contexts,
                    pi = FLAGS.policy, 
                    eps = FLAGS.eps, 
                    subset_r = FLAGS.subset_r) 
    else:
        raise NotImplementedError
    # sys.exit()

    if FLAGS.data_type == 'mnist': # Use 1000 test points for mnist 
        FLAGS.num_test_contexts = 1000  
        FLAGS.test_freq = 100
        FLAGS.chunk_size = 1
    
    # returned dataset = (contexts, actions, rewards, test_contexts, mean_test_rewards) 
    # rewards are added with noise, while mean_(test_)rewards are pure rewards
    dataset = data.reset_data()
    # print(f'Contexts: {dataset[0]}')
    context_dim = dataset[0].shape[1] 
    print(f'context_dim: {context_dim}')
    # print(f'Actions: {dataset[1]}')
    # print(f'rewards: {dataset[2]}')
    # print(f'test_contexts: {dataset[3]}')
    # print(f'mean_test_rewards: {dataset[4]}')
    # sys.exit()
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
        'debug_mode': 'full' # simple/full
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

    data_prefix = '{}_d={}_a={}_pi={}_std={}'.format(FLAGS.data_type, \
            context_dim, num_actions, policy_prefix, data.noise_std)

    res_dir = os.path.join('results', data_prefix) 

    if not os.path.exists(res_dir):
        os.makedirs(res_dir)

       

    #================================================================
    # Algorithms 
    #================================================================

    if FLAGS.algo_group == 'approx-neural':
        algos = [
            # class UniformSampling(BanditAlgorithm)
                # UniformSampling(lin_hparams),
                # NeuralGreedyV2(hparams, update_freq = FLAGS.update_freq), 
                # class ApproxNeuraLCBV2(BanditAlgorithm)
                # ApproxNeuraLCBV2_cp(hparams, update_freq = FLAGS.update_freq)
                ApproxNeuraLCBV2(hparams, update_freq = FLAGS.update_freq)
            ]

        algo_prefix = 'approx-neural-gridsearch_epochs={}_m={}_layern={}_buffer={}_bs={}_lr={}_beta={}_lambda={}_lambda0={}'.format(
            hparams.num_steps, min(hparams.layer_sizes), hparams.layer_n, hparams.buffer_s, hparams.batch_size, hparams.lr, \
            hparams.beta, hparams.lambd, hparams.lambd0
        )

    
    if FLAGS.algo_group == 'neural-greedy':
        algos = [
                UniformSampling(lin_hparams),
                NeuralGreedyV2(hparams, update_freq = FLAGS.update_freq), 
            ]

        algo_prefix = 'neural-greedy-gridsearch_epochs={}_m={}_layern={}_buffer={}_bs={}_lr={}_lambda={}'.format(
            hparams.num_steps, min(hparams.layer_sizes), hparams.layer_n, hparams.buffer_s, hparams.batch_size, hparams.lr, \
           hparams.lambd
        ) 
      


    if FLAGS.algo_group == 'baseline':
        algos = [
            UniformSampling(lin_hparams),
            LinLCB(lin_hparams),
            ## KernLCB(lin_hparams), 
            # NeuralGreedyV2(hparams, update_freq = FLAGS.update_freq),
            # ApproxNeuralLinLCBV2(hparams), 
            # ApproxNeuralLinGreedyV2(hparams),
            NeuralLinGreedyJointModel(hparams), 
            ApproxNeuralLinLCBJointModel(hparams)

        ]

        algo_prefix = 'baseline_epochs={}_m={}_layern={}_beta={}_lambda0={}_rbf-sigma={}_maxnum={}'.format(
            hparams.num_steps, min(hparams.layer_sizes), hparams.layer_n, \
            hparams.beta, hparams.lambd0, lin_hparams.rbf_sigma, lin_hparams.max_num_sample
        )

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
    #==============================
    # Runner 
    #==============================
    # file path for saving the results
    file_name = os.path.join(res_dir, algo_prefix) + '.npz' 
    
    # this is the core function that run all the experiments
    print(f'starting contextual_bandit_runner() ......')
    start  =  time.time()
    regrets, errs = contextual_bandit_runner(algos, data, FLAGS.num_sim, FLAGS.update_freq, FLAGS.test_freq, FLAGS.verbose, FLAGS.debug, FLAGS.normalize, file_name)
 
    np.savez(file_name, regrets=regrets, errs=errs)
    print(f'total time for contextual bandit runner: {time.time()-start} seconds')


# thissetup is only executed only if the script is run directly from the command line, not when imported as a module in another python project scrpit.
# app() ensures that all the command-line arguments are parsed
if __name__ == '__main__': 
    app.run(main)

        