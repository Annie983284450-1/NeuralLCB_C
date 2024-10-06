import importlib
import pickle
import warnings

import time as time
import math
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.statespace.exponential_smoothing import ExponentialSmoothing
from statsmodels.tsa.statespace.dynamic_factor_mq import DynamicFactorMQ
from sklearn.linear_model import LogisticRegression
import numpy as np
import pandas as pd
import os
import tensorflow.keras as keras
from tensorflow.keras.models import Sequential, clone_model
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import EarlyStopping
import sys
import tensorflow as tf
tf.get_logger().setLevel('ERROR')
warnings.filterwarnings("ignore")
import multiprocessing 
import dill
import cp_funs.utils_cp as util
multiprocessing.get_context().Process().Pickle = dill
import importlib
import core.bandit_dataset
importlib.reload(core.bandit_dataset)
from core.bandit_dataset import BanditDataset
import copy

# if algorithms.neural_offline_bandit_cp import PT, PI cannot import algorithms.neural_offline_bandit_cp, there would be circular import error
# from algorithms.neural_offline_bandit_cp import ApproxNeuraLCBV2_cp, NeuralBanditModelV2, NeuralBanditModel



# This is how self.Ensemble_pred_interval_centers is calculated. 

# When 

# model.name.split('_')[1] == 'nn',

# boot_predictions[b] = model.out(model.params, np.r_[self.X_train, self.X_predict]).flatten() # for NeuralBanditModel 

# this is the case i am dealing with. It seems that now boo_predictions[b] is 2D instead of 1D compared to when model.name.split('_')[1] == 'nn2'. How could I change the code and deal with 

# boot_predictions[b] when model.name.split('_')[1] == 'nn'.


class prediction_interval():
    '''
        Create prediction intervals using different methods (i.e., EnbPI, J+aB ICP, Weighted, Time-series)
    '''
 

    def __init__(self,  nn_model,  # The neural network model (NeuralBanditModelV2)
                X_train, X_predict, Y_train, Y_predict,  actions, test_actions, filename, algoname):
        
        # self.nn_model = algo.nn  # NeuralBanditModelV2(opt, hparams, '{}-net'.format(name))
        self.nn_model= nn_model

        # reset_data() will return a form of (contexts, actions, rewards, test_contexts, mean_test_rewards) 
        # cmab = OfflineContextualBandit(*data.reset_data(sim))
        self.X_train =  X_train
        self.X_predict = X_predict
        self.Y_train = Y_train
        self.Y_predict = Y_predict
        self.actions = actions
        self.test_actions = test_actions
        self.algoname = algoname

        
        # list of models for B bootstraps
        self.Ensemble_fitted_func = []
        self.Ensemble_online_resid = np.array([])
        self.Ensemble_pred_interval_centers = []   
        
        # self.final_result_path = final_result_path
        self.Ensemble_train_interval_centers = []  # Predicted training data centers by EnbPI
        # self.Ensemble_pred_interval_centers = []  # Predicted test data centers by EnbPI
        # self.Ensemble_online_resid = np.array([])  # LOO scores
        # self.Ensemble_pred_interval_ends = []  # Upper and lower end
        self.beta_hat_bins = []
        # self.ICP_fitted_func = []  # it only store 1 fitted ICP func.
        # self.ICP_resid = np.array([])
        # self.WeightCP_online_resid = np.array([])
        # self.JaB_boot_samples_idx = 0
        # self.JaB_boot_predictions = 0
        self.filename = filename
        # self.sim = sim


    def fit_bootstrap_models_online(self,  B, miss_test_idx):
        '''
          Train B bootstrap estimators from subsets of (X_train, Y_train), compute aggregated predictors, and compute the residuals
        '''
        n = len(self.X_train)  
        n1 = len(self.X_predict)
        print('====================================Size Checking of fit_bootstrap_models_online()====================================:')
        print(f'~~~~~~~~~~self.X_train.shape === {self.X_train.shape}~~~~~~~~~~')
        print(f'~~~~~~~~~~self.X_predict.shape ==={self.X_predict.shape}~~~~~~~~~~')
        print(f'~~~~~~~~~~self.Y_train.shape === {self.Y_train.shape}~~~~~~~~~~')
        print(f'~~~~~~~~~~self.Y_predict.shape ==={self.Y_predict.shape}~~~~~~~~~~')
        boot_samples_idx = util.generate_bootstrap_samples(n,n,B)
        # hold predictions from each f^b, for the whole datatset
        boot_predictions = np.zeros((B, (n+n1)), dtype=float)
        # for i^th column, it shows which f^b uses i in training (so exclude in aggregation)
        in_boot_sample = np.zeros((B, n), dtype=bool)
        out_sample_predict = np.zeros((n, n1))
        print(f'        Calculating the residuals......')
        start = time.time()
        # Loop over bootstrap models
        for b in range(B):
            # Clone the current neural network model
            # model = clone_model(self.nn_model)
            model = self.nn_model.clone()
            tmp_data = BanditDataset(model.hparams.context_dim, model.hparams.num_actions, len(self.X_train), f'{b}_th_fitdata')
            # Add the bootstrapped data into the model


            # this is not correct, the actions are not valid. 
            tmp_data.add(self.X_train[boot_samples_idx[b], :], self.actions[boot_samples_idx[b]], self.Y_train[boot_samples_idx[b]])
            # Train the model on the bootstrapped dataset
            # print(f'data after added:{data}')
            # sys.exit()
            
            # dataset = (contexts, actions, rewards, test_contexts, mean_test_rewards)
            print(f'*********  {b}-th Bootstrap  ****************')
            print(f'tmp_data.contexts.shape:{tmp_data.contexts.shape}')
            print(f'tmp_data.rewards.shape:{tmp_data.rewards.shape}')


            # def train(self, data, num_steps)

            
            model.train(tmp_data, model.hparams.num_steps)
            # Predict using the trained model on the combined training and prediction set

            

            # def out_impure_fn(self, params, contexts, actions):

            # nn and nn2 are generating different shapes for predictions, they should be handled differently.
            # Instead of flattening, handle boot_predictions in a way that maintains the correct structure so that it can be used properly in subsequent computations.
            if model.name.split('_')[1] == 'nn2':
                boot_predictions[b] = model.out(model.params, np.r_[self.X_train, self.X_predict],  np.r_[self.actions, self.test_actions]).flatten() # for NeuralBanditModelV2
                print(f'..........boot_predictions[{b}.shape == {boot_predictions[b].shape}..........')
            elif model.name.split('_')[1] == 'nn':
                boot_predictions[b] = model.out(model.params, np.r_[self.X_train, self.X_predict]).flatten() # for NeuralBanditModel
                print(f'..........boot_predictions[{b}.shape == {boot_predictions[b].shape}..........')
          
       

            self.Ensemble_fitted_func.append(model)  # Save the model for later use
            in_boot_sample[b, boot_samples_idx[b]] = True

        # Leave-One-Out aggregation method: Handle residuals for LOO prediction
        # keep = []
        for i in range(n):
            b_keep = np.argwhere(~(in_boot_sample[:, i])).reshape(-1)
            if len(b_keep) > 0:
                # Aggregate the predictions by taking the mean of bootstraps that do not include sample i
                self.Ensemble_train_interval_centers.append(boot_predictions[b_keep, i].mean())
                resid_LOO = self.Y_train[i] - boot_predictions[b_keep, i].mean()
                
                out_sample_predict[i] = boot_predictions[b_keep, n:].mean(0)
            else:
                resid_LOO = self.Y_train[i]  
                out_sample_predict[i] = np.zeros(n1)

            # Update the residuals for online conformal prediction
            self.Ensemble_online_resid = np.append(self.Ensemble_online_resid, resid_LOO)
            # keep = keep+[]
        print(f'Max LOO training residual is {np.max(self.Ensemble_online_resid)}')
        print(f'Min LOO training residual is {np.min(self.Ensemble_online_resid)}')
        # Final step: Update for residuals in the prediction set
        sorted_out_sample_predict = out_sample_predict.mean(axis=0)
        resid_out_sample = self.Y_predict - sorted_out_sample_predict # length n1
        print(f'                       resid_out_sample.shape === {resid_out_sample.shape}')

        if len(miss_test_idx) > 0:
            for l in range(len(miss_test_idx)):
                i = miss_test_idx[l]
                if i > 0:
                    j = i - 1
                    while j in miss_test_idx[:l]:
                        j -= 1
                    resid_out_sample[i] = resid_out_sample[j]
                else:
                    resid_out_sample[0] = self.Ensemble_online_resid[-1]

        # Update the residuals with the new out-of-sample residuals
        self.Ensemble_online_resid = np.append(self.Ensemble_online_resid, resid_out_sample)
        self.Ensemble_pred_interval_centers = sorted_out_sample_predict
        return self.Ensemble_pred_interval_centers # length n1



    def compute_PIs_Ensemble_online(self, alpha, stride):
        print(f'                        Running compute_PIs_Ensemble_online(alpha=0.05, stride=10)-----------')
        n = len(self.X_train)
        n1 = len(self.Y_predict)
        # Now f^b and LOO residuals have been constructed from earlier using fit_bootstrap_models_online(B, miss_test_idx)
        out_sample_predict = self.Ensemble_pred_interval_centers
        start = time.time()
        '''
        def strided_app(a, L, S):  # Window len = L, Stride len/stepsize = S
            nrows = ((a.size - L) // S) + 1
            n = a.strides[0]
            return np.lib.stride_tricks.as_strided(a, shape=(nrows, L), strides=(S * n, n))
        L: The length of each window. This is how many consecutive elements are included in each row of the resulting matrix.
        S: The stride or step size. This determines how far you move forward in the array to start the next window.
        '''
        resid_strided = util.strided_app(self.Ensemble_online_resid[:-1], n, stride)
        num_unique_resid = resid_strided.shape[0]
        print('     num_unique_resid:', num_unique_resid)
        print(f'        size of resid_strided: {resid_strided.shape}')
        # print(f'        resid_strided:{resid_strided}')

        width_left = np.zeros(num_unique_resid)
        width_right = np.zeros(num_unique_resid)
        for i in range(num_unique_resid):
            # past_resid = ri, r_{i+1},..., r_{i+n-1}
            # traverse each row
            past_resid = resid_strided[i, :]
            # The number of bins will be determined INSIDE binning, i.e., 5

            # util.binning will minimize the width 
            beta_hat_bin = util.binning(past_resid, alpha)
            self.beta_hat_bins.append(beta_hat_bin)
            width_left[i] = np.percentile(past_resid, math.ceil(100*beta_hat_bin))
            width_right[i] = np.percentile(past_resid, math.ceil(100*(1-alpha+beta_hat_bin)))

        print(
            f'~~~~~~~~~~~~Finish Computing {num_unique_resid} UNIQUE Prediction Intervals, took {time.time()-start} secs.~~~~~~~~~~~~')
        # repeat the nd array for stride times
        # herein, we set stride = 1

        width_left = np.repeat(width_left, stride)  # This is because |width|=T1/stride.
        width_right = np.repeat(width_right, stride)  # This is because |width|=T1/stride.
        print("size of width_left:", width_left.size)
 

        # n1X2 data frame
        print(f'            out_sample_predict.shape:{out_sample_predict.shape}')
        PIs_Ensemble = pd.DataFrame(np.c_[out_sample_predict+width_left,
                                          out_sample_predict+width_right], columns=['lower', 'upper'])
        self.Ensemble_pred_interval_ends = PIs_Ensemble
        print(f'                        ~~~~~~~~~PIs_Ensemble.shape === {PIs_Ensemble.shape}')
        # print(time.time()-start)
        return PIs_Ensemble




# main function
    def run_experiments(self, alpha, stride,  methods=['Ensemble']):
        """
        Run conformal prediction experiments.
        Args:
            alpha: Confidence level for the prediction intervals.
            stride: Stride for prediction intervals.
            data_name: Dataset name for experiment.
            itrial: Trial number.
            true_Y_predict: Ground truth for prediction (optional).
            methods: Methods to be used for conformal prediction.
        """
        PIs = []
        results = pd.DataFrame(columns=[ 'train_size', 'mean_coverage', 'avg_width', 'mean_lower', 'mean_upper'])
        for method in methods:
            if method == 'Ensemble':
                PI = self.compute_PIs_Ensemble_online(alpha, stride)
                PI['method'] = method
            else:
                raise NotImplementedError(f"Method {method} not implemented.")
            
            PI['alpha'] = alpha
             
            PIs.append(PI)

            # Calculate coverage and width
            # print(' =====. Debugging =====. ')
            print("Shape of PI['lower']:", PI['lower'].shape)
            print("Shape of PI['upper']:", PI['upper'].shape)
            print("Shape of self.Y_predict:", self.Y_predict.shape)

            mean_coverage = ((PI['lower'] <= self.Y_predict) & (PI['upper'] >= self.Y_predict)).mean()
            mean_width = (PI['upper'] - PI['lower']).mean()
            lower_mean = PI['lower'].mean()
            upper_mean = PI['upper'].mean()

            results.loc[len(results)] = [len(self.X_train), mean_coverage, mean_width, lower_mean, upper_mean]
            final_result_path = self.filename
            new_row_all_avg = results
            if not isinstance(new_row_all_avg, pd.DataFrame):
                new_row_all_avg = pd.DataFrame([new_row_all_avg])
            with open(final_result_path+f'/final_all_cpresults_avg_{self.algoname}.csv', 'a') as f:
                new_row_all_avg.to_csv(f, header=f.tell()==0, index=False)
            print(f'~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
            print(f'Conformal Prediction Results:{results}')
            print(f'~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')

        return pd.concat(PIs, axis=1), results


     