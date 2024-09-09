'''
the prediction interval class specifically designed for
NeuralLCB
'''

import importlib
import pickle
import warnings
import utils_Sepsysolcp as util
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
import utils_Sepsysolcp as util
from core.bandit_dataset import BanditDataset


class prediction_interval():
    def __init__(self, nn_model, X_train, X_predict, Y_train, Y_predict):
        self.nn_model = nn_model
        self.X_train = X_train
        self.X_predict = X_predict
        self.Y_train = Y_train
        self.Y_predict = Y_predict
        self.Ensemble_fitted_func = []
        self.Ensemble_online_resid = np.array([])
        # self.precomputed_preds = precomputed_preds

        # Predicted test data centers by EnbPI
        self.Ensemble_pred_interval_centers = []

    def fit_bootstrap_models_online(self, alpha, B, miss_test_idx):
        n = len(self.X_train)
        n1 = len(self.X_predict)
        boot_samples_idx = util.generate_bootstrap_samples(n,n,B)
        boot_predictions = np.zeros((B,n+n1),dtype=float)
        in_boot_sample = np.zeros((B,n),dtype=bool)
        out_sample_predict = np.zeros((n,n1))
        # ind_q = int((1-alpha)*n)

        for b in range(B):
            model = self.nn_model
            data = BanditDataset(model.hparams.context_dim, model.hparams.num_actions, len(self.X_train), 'fit-data')
            # BanditDataset.add(context, action, reward)
            # maybe actions are not used herein??????????
            data.add(self.X_train[boot_samples_idx[b], :], np.zeros(len(boot_samples_idx[b])), self.Y_train[boot_samples_idx[b]])
            model.train(data, model.hparams.num_steps)
            boot_predictions[b] = model.out(model.params, np.r_[self.X_train, self.X_predict], np.zeros((n + n1,))).flatten()
            self.Ensemble_fitted_func.append(model)
            in_boot_sample[b, boot_samples_idx[b]] = True
        # leave one out aggregation method
        # loop over n training samples
        for i in range(n):
            b_keep = np.argwhere(~(in_boot_sample[:,i])).reshape(-1)
            if len(b_keep)>0:
                # aggregate the bootstraps that do not include sample i, use the mean as the predicted Y
                resid_LOO = np.abs(self.Y_train[i] - boot_predictions[b_keep,i].mean())
                # self.Ensemble_online_resid = np.append(self.Ensemble_online_resid, resid_LOO)
                # leave the i-th sample out based predictions of Y_predict
                out_sample_predict[i] = boot_predictions[b_keep, n:].mean(0)

            else: # len(b_keep)=0:
                resid_LOO = np.abs(self.Y_train[i])
                out_sample_predict[i] = np.zeros(n1)
            

            
            self.Ensemble_online_resid = np.append(self.Ensemble_online_resid, resid_LOO)
            keep = keep + []

        # sorted_out_sample_predict = np.sort(out_sample_predict, axis=0)[ind_q]
        # Then we calculate the means of Y_predict for each Y_predict 
        sorted_out_sample_predict = out_sample_predict.mean(axis=0)  # length n1 
        resid_out_sample = self.Y_predict-sorted_out_sample_predict
   

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
        self.Ensemble_online_resid = np.append(self.Ensemble_online_resid, resid_out_sample)
        self.Ensemble_pred_interval_centers = sorted_out_sample_predict
        return sorted_out_sample_predict
    

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
            # if i <= 5:
            #     print(f'Beta hat bin at {i+1}th prediction index is {beta_hat_bin}')
            #     print(f'Lower end is {width_left[i]} \n Upper end is {width_right[i]}')
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


    def run_experiments(self, alpha, stride, data_name, itrial, true_Y_predict=[],   none_CP=False, methods=['Ensemble', 'ICP', 'Weighted_ICP']):
        '''
        NOTE: I added a "true_Y_predict" option, which will be used for calibrating coverage under missing data
            In particular, this is needed when the Y_predict we use for training is NOT the same as true Y_predict
            Anni Zhou Feb 22, 2023: So, we do not need that true_Y_predict in "hours to sepsis prediction"
        '''
        train_size = len(self.X_train)
        np.random.seed(98765+itrial)
        # this is for one dim case, skip for now
        if none_CP: 
            results = pd.DataFrame(columns=['itrial', 'dataname',
                                            'method', 'train_size', 'coverage', 'width'])
            print('Not using Conformal Prediction Methods')
            save_name = {'ARIMA(10,1,10)': 'ARIMA',
                         'ExpSmoothing': 'ExpSmoothing',
                         'DynamicFactor': 'DynamicFactor'}
            PIs = []
             
            for name in save_name.keys():
                print(f'Running {name}')
                ## only use Y to predict, one_dim  = true
                PI_res = self.compute_PIs_tseries_online(alpha, name=name)
                mean_coverage_res = ((np.array(PI_res['lower']) <= self.Y_predict) & (
                    np.array(PI_res['upper']) >= self.Y_predict)).mean()
                print(f'Average Coverage is {mean_coverage_res}')
                mean_width_res = (PI_res['upper'] - PI_res['lower']).mean()
                print(f'Average Width is {mean_width_res}')
                results.loc[len(results)] = [itrial, data_name, save_name[name],
                                             train_size, mean_coverage_res, mean_width_res]
                PIs.append(PI_res)
        else: # Conformal case
            results = pd.DataFrame(columns=['itrial', 'dataname', 'muh_fun',
                                   'method', 'train_size', 'mean_coverage', 'avg_width','mean_lower', 'mean_upper'])
            PIs = []
            for method in methods:
                print(f'**********Runnning {method} ......')
                if method == 'JaB':  
                    PI = self.compute_PIs_JaB(alpha)
                    PI['method'] = method
                elif method == 'Ensemble': # focus on this one
                    # methods = ['Ensemble', 'ICP', 'Weighted_ICP']s
                    # PI: n1X2
                    PI = eval(f'compute_PIs_{method}_online({alpha},{stride})',
                              globals(), {k: getattr(self, k) for k in dir(self)})
                else:
                    # for ICP and weighted ICP, we have 50% of the dataset as training dataset
                    l = math.ceil(0.5*len(self.X_train))
                    # compute_PIs_Ensemble_online(self, alpha, stride)
                    # The globals() method returns a dictionary with all the global variables and symbols for the current program.
                    # PI returns the lower and upper bound of each entry in the predeiction set
                    # methods=['Ensemble', 'ICP', 'Weighted_ICP']
                    PI = eval(f'compute_PIs_{method}({alpha},{l})',
                              globals(), {k: getattr(self, k) for k in dir(self)})
              

                # PI['lower'] = [max_hours+1 if y > max_hours+1 else y for y in PI['lower']]
                # PI['lower'] = [0 if y<0 else y for y in PI['lower']]
                # PI['upper'] = [max_hours+1 if y > max_hours+1 else y for y in PI['upper']]
                # PI['upper'] = [0 if y<0 else y for y in PI['upper']] 
                PI['method'] = method 
                PI['alpha'] = alpha    
                PI['itrial'] = itrial

                PIs.append(PI)
                # evaluate the coverage of all testing dataset
                mean_coverage = ((np.array(PI['lower']) <= self.Y_predict) & (
                    np.array(PI['upper']) >= self.Y_predict)).mean()
                # skip this case
                if len(true_Y_predict) > 0:
                    mean_coverage = ((np.array(PI['lower']) <= true_Y_predict) & (
                        np.array(PI['upper']) >= true_Y_predict)).mean()
                print(f'Average Coverage is {mean_coverage}')
                # width is based on the average value
                mean_width = (PI['upper'] - PI['lower']).mean()
                lower_mean = PI['lower'].mean()
                upper_mean = PI['upper'].mean()
                print(f'Average Width is {mean_width}')
                print('-------------------------------------')
                # add to the end of the dataframe
                # the results contains the average value, but what I want might be the accurate confidence interval on an hourly basis?
                # results.loc[len(results)] = [itrial, data_name,
                #                              self.regressor.__class__.__name__, method, train_size, mean_coverage, mean_width, lower_mean, upper_mean]         
                results.loc[len(results)] = [itrial, data_name,
                                             self.nn_model.__class__.__name__, method, train_size, mean_coverage, mean_width, lower_mean, upper_mean]          
            PIs_df = pd.concat(PIs, axis=1)
        return PIs_df, results 