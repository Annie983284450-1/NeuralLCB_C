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
    def __init__(self, nn_model, X_train, X_predict, Y_train, Y_predict, precomputed_preds = None):
        self.nn_model = nn_model
        self.X_train = X_train
        self.X_predict = X_predict
        self.Y_train = Y_train
        self.Y_predict = Y_predict
        self.Ensemble_fitted_func = []
        self.Ensemble_online_resid = np.array([])
        self.precomputed_preds = precomputed_preds

    def fit_bootstrap_models_online(self, alpha, B, miss_test_idx):
        n = len(self.X_train)
        n1 = len(self.X_predict)
        boot_samples_idx = util.generate_bootstrap_samples(n,n,B)
        boot_predictions = np.zeros((B,n+n1),dtype=float)
        in_boot_sample = np.zeros((B,n),dtype=bool)
        out_sample_predict = np.zeros((n,n1))
        ind_q = int((1-alpha)*n)

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
                self.Ensemble_online_resid = np.append(self.Ensemble_online_resid, resid_LOO)


   



