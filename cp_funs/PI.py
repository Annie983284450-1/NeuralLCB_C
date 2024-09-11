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
multiprocessing.get_context().Process().Pickle = dill
from core.bandit_dataset import BanditDataset



class prediction_interval():
    '''
        Create prediction intervals using different methods (i.e., EnbPI, J+aB ICP, Weighted, Time-series)
    '''
 

    def __init__(self, fit_func, X_train, X_predict, Y_train, Y_predict, final_result_path):
        
        self.regressor = fit_func
        self.X_train = X_train
        self.X_predict = X_predict
        self.Y_train = Y_train
        self.Y_predict = Y_predict
        self.final_result_path = final_result_path
        self.Ensemble_train_interval_centers = []  # Predicted training data centers by EnbPI
        self.Ensemble_pred_interval_centers = []  # Predicted test data centers by EnbPI
        self.Ensemble_online_resid = np.array([])  # LOO scores
        self.Ensemble_pred_interval_ends = []  # Upper and lower end
        self.beta_hat_bins = []
        self.ICP_fitted_func = []  # it only store 1 fitted ICP func.
        self.ICP_resid = np.array([])
        self.WeightCP_online_resid = np.array([])
        self.JaB_boot_samples_idx = 0
        self.JaB_boot_predictions = 0

    def fit_bootstrap_models_online(self, B, miss_test_idx, Isrefit, model_name, max_hours):
        '''
          Train B bootstrap estimators from subsets of (X_train, Y_train), compute aggregated predictors, and compute the residuals
        '''
        n = len(self.X_train)  
        n1 = len(self.X_predict)  
 
        saved_model_path =  './saved_models/'+ model_name 
        if not os.path.exists(saved_model_path):
 
            os.makedirs(saved_model_path)
        if Isrefit:
            
            print(f'        !!!!!Refitting {model_name}!!!!!')
            boot_samples_idx = util.generate_bootstrap_samples(n, n, B) 
      
            np.save(os.path.join(saved_model_path,'boot_samples_idx.npy'), boot_samples_idx)
        else:
            print(f'        No refitting, use saved {model_name} models!!!')
            boot_samples_idx = np.load(os.path.join(saved_model_path,'boot_samples_idx.npy'))
            # print(f'        boot_samples_idx: {boot_samples_idx}')
        # print(f'        Size of $boot_samples_idx$: {boot_samples_idx.shape}')
        # hold predictions from each f^b, for the whole datatset
        boot_predictions = np.zeros((B, (n+n1)), dtype=float)
        # for i^th column, it shows which f^b uses i in training (so exclude in aggregation)
        in_boot_sample = np.zeros((B, n), dtype=bool)
        out_sample_predict = np.zeros((n, n1))
        print(f'        Calculating the residuals......')
        start = time.time()
        # S_b
        # the boot strap is trained on the training dataset only

        for b in range(B):
            # only refit the model when we need to do
            #i.e., the first fitting, and every 100 patients
            if Isrefit:
                model = self.regressor
                # NOTE: it is CRITICAL to clone the model, as o/w it will OVERFIT to the model across different iterations of bootstrap S_b.
                # I originally did not understand that doing so is necessary but now know it
                if self.regressor.__class__.__name__ == 'Sequential':
                    start1 = time.time()
                    model = clone_model(self.regressor)
                    opt = Adam(0.0005)
                    model.compile(loss='mean_squared_error', optimizer=opt)
                    callback = keras.callbacks.EarlyStopping(monitor='loss', patience=10)
                    bsize = int(0.1*len(np.unique(boot_samples_idx[b])))
                    # 
                    if self.regressor.name == 'NeuralNet':
                        # verbose definition here: https://keras.io/api/models/model_training_apis/#fit-method. 0 means silent
                        # NOTE: I do NOT want epoches to be too large, as we then tend to be too close to the actual Y_t, NOT f(X_t).
                        # Originally, epochs=1000, batch=100
                        model.fit(self.X_train[boot_samples_idx[b], :], self.Y_train[boot_samples_idx[b], ],
                                epochs=250, batch_size=bsize, callbacks=[callback], verbose=0)
                
                    else:
                        # This is RNN, mainly have different shape and decrease epochs for faster computation
                        model.fit(self.X_train[boot_samples_idx[b], :], self.Y_train[boot_samples_idx[b], ],
                                epochs=10, batch_size=bsize, callbacks=[callback], verbose=0)
                    filename = 'No_'+str(b)+'th_boot'+'.h5'
                    saved_model_file_name = os.path.join(saved_model_path, filename)
                    model.save(saved_model_file_name)              
                    # NOTE, this multiplied by B tells us total estimation time
                    # print(f'Took {time.time()-start1} secs to fit the {b}th boostrap model------------')
                else: # no NN models
                    model.fit(self.X_train[boot_samples_idx[b], :],
                                    self.Y_train[boot_samples_idx[b], ])
                    filename = 'No_'+str(b)+'th_boot'+'.pkl'
                    saved_model_file_name = os.path.join(saved_model_path, filename)
                    # if already exists, 'wb' gives the access to overwrite the previous boot
                    with open(saved_model_file_name,'wb') as f:
                        pickle.dump(model,f)    
            else: #Isrefit = False
                # Do not refit, just use the old fitted model to do all the predictions
                # print(f'No fitting, use the previous fitted model !!!!!!!!!')
                # read the models fitted previously
                if self.regressor.__class__.__name__ == 'Sequential':
                    saved_model_path = './saved_models/'+ model_name 
                    saved_filename = 'No_'+str(b)+'th_boot'+'.h5'
                    saved_model_file_name = os.path.join(saved_model_path, saved_filename)
                    model = load_model(saved_model_file_name)
                else:
                    saved_model_path =  './saved_models/'+ model_name 
                    saved_filename = 'No_'+str(b)+'th_boot'+'.pkl'
                    saved_model_file_name = os.path.join(saved_model_path, saved_filename)
                    with open(saved_model_file_name, 'rb') as f:
                        model = pickle.load(f)         
              
            # np.r_: Translates slice objects to concatenation along the first axis.
            # calculate mu_hat_b for the whole dataset for both trainning and predicting dataset
            boot_predictions[b] = model.predict(np.r_[self.X_train, self.X_predict]).flatten() # to one dimension
            in_boot_sample[b, boot_samples_idx[b]] = True
        print(f'        ///Finish Fitting {B} Bootstrap for "{model_name}", took {time.time()-start} secs.///')

        start = time.time()
        keep = []
        ## the aggregation method of reusing the mu_hat that have been got in the B bootstrap training above
        num_null_bkeep=0
        for i in range(n):
            # in_boot_sample = np.zeros((B, n), dtype=bool)
            # it is possible that np.random.choice(n, m) can produce repetitive values
            # aggregate only the mu_hat_b whose underlying training data set S_b did not include the i-th data point
            
            b_keep = np.argwhere(~(in_boot_sample[:, i])).reshape(-1) # the idexes of bootstrap that are used to calculate the mu_hat_-i
            # if i==n or i/10==0:
            #     # print(f'{i}th training sample...........')
            #     print(f'b_keep for {i}th training sample: {b_keep}')
            if(len(b_keep) > 0): # there exists at least one boot strap whose S_b did not include the i-th data point
                 # NOTE: Append these training centers to see their magnitude
                # The reason is sometimes they are TOO close to actual Y.
                # get the mean of all the training samples that did not include the i-th data point, appended as one of the centers
                self.Ensemble_train_interval_centers.append(boot_predictions[b_keep, i].mean())
                ## residual = y_train_i - mu_hat_-i
                # resid_LOO can be a negative number
                resid_LOO = self.Y_train[i] - boot_predictions[b_keep, i].mean()
                ##  each (X_predict Y_predict) is calculated as the mean of all the bootstraps that did not include i-th data point
                # boot_predictions[b] = model.predict(np.r_[self.X_train, self.X_predict]).flatten()
                # calculate the mean of each column on all the rows
                # get the mean value of all the bootstraps that do not use X_train[i] during training
                # and only include predicted Y of thet testing set
                out_sample_predict[i] = boot_predictions[b_keep, n:].mean(0)
                keep = keep+[b_keep]
            else:  # if we cannot find a boot strap that did not include i-th data point, let mu_hat_-i ==0
                # print(f'$len(b_keep)==0$ !!!')
                resid_LOO = self.Y_train[i] #-0
                out_sample_predict[i] = np.zeros(n1) # let all Y_predict == 0 
                num_null_bkeep = num_null_bkeep+1
            self.Ensemble_online_resid = np.append(
                self.Ensemble_online_resid, resid_LOO)
            keep = keep+[] 
        # print(f'###Max LOO training residual is {np.max(self.Ensemble_online_resid)}')
        # print(f'###Min LOO training residual is {np.min(self.Ensemble_online_resid)}')
        print(f'        There are #{num_null_bkeep}/{n} training sample that do not have LOO bootstraps!!!')
        # out_sample_predict = np.zeros((n, n1))
        # out_sample_predict[i] = boot_predictions[b_keep, n:].mean(0)
        # recall that each (X_predict Y_predict) is calculated as the mean of all the bootstraps that did not include i-th data point
        # each column of out_sample_predict(n, n1) is the mean of B bootstaps, there is n means for each Y_predict, because we run for n loops as what's shown above
        # Then we calculate the means of Y_predict for each Y_predict 
        sorted_out_sample_predict = out_sample_predict.mean(axis=0)  # length n1 
        # limit the predictions to [0, 500]
        sorted_out_sample_predict = [max_hours+1 if y > max_hours+1  else y for y in sorted_out_sample_predict]
        sorted_out_sample_predict = [0 if y <0 else y for y in sorted_out_sample_predict]
        # print(f'$$$Size of sorted_out_sample_predict: {sorted_out_sample_predict.shape}')
        # It is the similar way as how we get the resid_loo, the only difference is 
        # for resid_loo, we use the mean from B bootstraps
        # for resid_out_sample, we use the mean from n "bootstraps". The new n "bootstraps" are automatically generated when we are doing a n loop 
        # to find the bootstraps that did not contain i-th data point during training
        resid_out_sample = self.Y_predict-sorted_out_sample_predict
        # usually miss_test_idx = [], skip for now
        if len(miss_test_idx) > 0:
            # Replace missing residuals with that from the immediate predecessor that is not missing, as
            # o/w we are not assuming prediction data are missing
            for l in range(len(miss_test_idx)):
                i = miss_test_idx[l]
                if i > 0:
                    j = i-1
                    while j in miss_test_idx[:l]:
                        j -= 1
                    resid_out_sample[i] = resid_out_sample[j]

                else:
                    # The first Y during testing is missing, let it be the last of the training residuals
                    # note, training data already takes out missing values, so doing it is fine
                    resid_out_sample[0] = self.Ensemble_online_resid[-1]
        # len(Ensemble_online_resid)  == n+n1
        self.Ensemble_online_resid = np.append(
            self.Ensemble_online_resid, resid_out_sample)
        print(f'        ~~~~~~Finish Computing LOO residuals for "{model_name}", took {time.time()-start} secs.~~')
        # print(f'        ###Max LOO test residual is {np.max(self.Ensemble_online_resid[n:])}')
        # print(f'        ###Min LOO test residual is {np.min(self.Ensemble_online_resid[n:])}')
        print('\n')
        # sorted_out_sample_predict = out_sample_predict.mean(axis=0)  # length n1
        # self.Ensemble_pred_interval_centers is the predicted label using LOO techniques
        self.Ensemble_pred_interval_centers = sorted_out_sample_predict
        return model 

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




# main function
    def run_experiments(self, alpha, stride, data_name, itrial, true_Y_predict=[], get_plots=False, none_CP=False, methods=['Ensemble', 'ICP', 'Weighted_ICP'], max_hours=48):
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
                # limit the interval to [0, 500]
                PI['lower'] = [max_hours+1 if y > max_hours+1 else y for y in PI['lower']]
                PI['lower'] = [0 if y<0 else y for y in PI['lower']]
                PI['upper'] = [max_hours+1 if y > max_hours+1 else y for y in PI['upper']]
                PI['upper'] = [0 if y<0 else y for y in PI['upper']] 
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
                results.loc[len(results)] = [itrial, data_name,
                                             self.regressor.__class__.__name__, method, train_size, mean_coverage, mean_width, lower_mean, upper_mean]               
            PIs_df = pd.concat(PIs, axis=1)
        return PIs_df, results 
        # return PIs_df  

# if __name__=="__main__":


     
