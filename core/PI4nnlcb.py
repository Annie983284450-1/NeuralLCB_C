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
        



   



