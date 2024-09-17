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
from core.bandit_dataset import BanditDataset
import copy

class prediction_interval():
    '''
        Create prediction intervals using different methods (i.e., EnbPI, J+aB ICP, Weighted, Time-series)
    '''
 

    def __init__(self, nn_model, X_train, X_predict, Y_train, Y_predict):
        
        self.nn_model = nn_model
        self.X_train = X_train
        self.X_predict = X_predict
        self.Y_train = Y_train
        self.Y_predict = Y_predict
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



    def fit_bootstrap_models_online(self,  B, miss_test_idx):
        '''
          Train B bootstrap estimators from subsets of (X_train, Y_train), compute aggregated predictors, and compute the residuals
        '''
        n = len(self.X_train)  
        n1 = len(self.X_predict)  
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
            data = BanditDataset(model.hparams.context_dim, model.hparams.num_actions, len(self.X_train), f'{b}_th_fitdata')
            # Add the bootstrapped data into the model
            data.add(self.X_train[boot_samples_idx[b], :], np.zeros(len(boot_samples_idx[b])), self.Y_train[boot_samples_idx[b]])
            # Train the model on the bootstrapped dataset
            # print(f'data after added:{data}')
            # sys.exit()
            
            # dataset = (contexts, actions, rewards, test_contexts, mean_test_rewards)
            print(f'data.contexts.shape:{data.contexts.shape}')
            model.train(data, model.hparams.num_steps)
            # Predict using the trained model on the combined training and prediction set
            boot_predictions[b] = model.out(model.params, np.r_[self.X_train, self.X_predict], np.zeros((n + n1,))).flatten() # for V2
          
       

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
        n = len(self.X_train)
        n1 = len(self.Y_predict)
        # Now f^b and LOO residuals have been constructed from earlier using fit_bootstrap_models_online(B, miss_test_idx)
        out_sample_predict = self.Ensemble_pred_interval_centers
        start = time.time()
        resid_strided = util.strided_app(self.Ensemble_online_resid[:-1], n, stride)
        num_unique_resid = resid_strided.shape[0]
        print('num_unique_resid:', num_unique_resid)
        print(f'size of resid_strided: {resid_strided.shape}')

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
        # store the lower and upper bound of each entry (prediction set only)

        # len(out_sample_predict) = n1 = len(Y_predict)
        # and we have nrows = floor(n1//stride)+1 = n1/stride
        # then len(width_left)  = len(width_right) = n1 = len(out_sample_predict)
        # herein, we need to make sure that n1/stride ==0, so that len(width_left)  = len(width_right) = n1 = len(out_sample_predict) with 100%??? Added by Annie Zhou Feb 23rd, 2023
        # width_left might be negative numbers, that's normal

        # n1X2 data frame
        PIs_Ensemble = pd.DataFrame(np.c_[out_sample_predict+width_left,
                                          out_sample_predict+width_right], columns=['lower', 'upper'])
        self.Ensemble_pred_interval_ends = PIs_Ensemble
        # print(time.time()-start)
        return PIs_Ensemble




# main function
    def run_experiments(self, alpha, stride, data_name, itrial, true_Y_predict=[], get_plots=False, none_CP=False, methods=['Ensemble', 'ICP', 'Weighted_ICP'], max_hours=48):
  
        """
        Compute the prediction intervals for online conformal prediction using Ensemble.
        Args:
            alpha: Confidence level (e.g., 0.05).
            stride: Stride length for processing.
        """
        n = len(self.X_train)
        n1 = len(self.Y_predict)

        # Use the residuals from earlier to compute the bounds
        out_sample_predict = self.Ensemble_pred_interval_centers
        resid_strided = util.strided_app(self.Ensemble_online_resid[:-1], n, stride)
        num_unique_resid = resid_strided.shape[0]

        width_left = np.zeros(num_unique_resid)
        width_right = np.zeros(num_unique_resid)

        for i in range(num_unique_resid):
            past_resid = resid_strided[i, :]
            beta_hat_bin = util.binning(past_resid, alpha)
            width_left[i] = np.percentile(past_resid, math.ceil(100 * beta_hat_bin))
            width_right[i] = np.percentile(past_resid, math.ceil(100 * (1 - alpha + beta_hat_bin)))

        width_left = np.repeat(width_left, stride)
        width_right = np.repeat(width_right, stride)

        # Store the lower and upper bounds for prediction intervals
        PIs_Ensemble = pd.DataFrame(np.c_[out_sample_predict + width_left, out_sample_predict + width_right], columns=['lower', 'upper'])
        self.Ensemble_pred_interval_ends = PIs_Ensemble
        return PIs_Ensemble

    def run_experiments(self, alpha, stride, data_name, itrial, true_Y_predict=[], methods=['Ensemble']):
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
        results = pd.DataFrame(columns=['itrial', 'dataname', 'method', 'train_size', 'mean_coverage', 'avg_width', 'mean_lower', 'mean_upper'])
        for method in methods:
            if method == 'Ensemble':
                PI = self.compute_PIs_Ensemble_online(alpha, stride)
                PI['method'] = method
            else:
                raise NotImplementedError(f"Method {method} not implemented.")
            
            PI['alpha'] = alpha
            PI['itrial'] = itrial
            PIs.append(PI)

            # Calculate coverage and width
            mean_coverage = ((PI['lower'] <= self.Y_predict) & (PI['upper'] >= self.Y_predict)).mean()
            mean_width = (PI['upper'] - PI['lower']).mean()
            lower_mean = PI['lower'].mean()
            upper_mean = PI['upper'].mean()

            results.loc[len(results)] = [itrial, data_name, method, len(self.X_train), mean_coverage, mean_width, lower_mean, upper_mean]

        return pd.concat(PIs, axis=1), results


     
