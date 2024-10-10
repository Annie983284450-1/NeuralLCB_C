"""Define neural offline bandit algorithms. """

import math
import pandas as pd
import jax 
import jax.numpy as jnp
import optax
import numpy as np 
from joblib import Parallel, delayed
from core.bandit_algorithm import BanditAlgorithm 
from core.bandit_dataset import BanditDataset
from tensorflow.keras.models import Sequential, clone_model
from core.utils import inv_sherman_morrison, inv_sherman_morrison_single_sample, vectorize_tree
import importlib
import algorithms.neural_bandit_model  # Import the entire module
# Reload the module
importlib.reload(algorithms.neural_bandit_model)
from algorithms.neural_bandit_model import NeuralBanditModel, NeuralBanditModelV2
import importlib
import cp_funs.PI
importlib.reload(cp_funs.PI)
from cp_funs.PI import prediction_interval
import importlib
import warnings
import time as time
import math
import numpy as np
import pandas as pd
import tensorflow as tf
tf.get_logger().setLevel('ERROR')
warnings.filterwarnings("ignore")
import multiprocessing 
import dill
 
multiprocessing.get_context().Process().Pickle = dill
import importlib
import core.bandit_dataset
importlib.reload(core.bandit_dataset)
from core.bandit_dataset import BanditDataset
# if algorithms.neural_offline_bandit_cp import PT, PI cannot import algorithms.neural_offline_bandit_cp, there would be circular import error
# from algorithms.neural_offline_bandit_cp import ApproxNeuraLCBV2_cp, NeuralBanditModelV2, NeuralBanditModel
# ======================================================================
# ========================================================================
# the ApproxNeuraLCB with conformal prediction
class ApproxNeuraLCB_cp(BanditAlgorithm):

    def __init__(self, hparams, res_dir, B, update_freq=1, name='ApproxNeuraLCB_cp'):
        self.name = name 
        self.hparams = hparams 
        self.update_freq = update_freq
        # learning rate 1e-3
        opt = optax.adam(hparams.lr)
        self.nn = NeuralBanditModelV2(opt, hparams, '{}_nn2'.format(name))
      
        # data buffer for incoming data, update each round when we have a new (c,a, r)
        self.data = BanditDataset(hparams.context_dim, hparams.num_actions, hparams.buffer_s, '{}-data'.format(name))
        self.diag_Lambda = [jnp.ones(self.nn.num_params) * hparams.lambd0 for _ in range(hparams.num_actions)]
        self.prediction_interval_model = None
        self.res_dir  = res_dir
        self.Ensemble_pred_interval_centers = [] 
        self.B =B  
    def reset(self, seed): 
        self.diag_Lambda = [jnp.ones(self.nn.num_params) * self.hparams.lambd0 for _ in range(self.hparams.num_actions)]
        # self.nn.reset(seed) # with NeuralBanditV2()
        self.data.reset()
        # print(f'~~~~~~~!!!!!! After running {self.name}.reset()~!!!!!!!!!!~~~~~~~~~~')
        # print(f'self.data.rewards.shape:{self.data.rewards}')
        # self.prediction_interval_model = None
    # line 5 in NeuraLCB Bmode
    # here is where conformal prediction and NeuraLCB integrated together

    def sample_action(self, test_contexts, opt_vals, opt_actions):
        # flags.DEFINE_integer('chunk_size', 500, 'Chunk size')
        # flags.DEFINE_integer('batch_size', 32, 'Batch size')
        cs = self.hparams.chunk_size
        # loo_preds = self.get_loo_params(contexts, actions, rewards)
        # print(f'LOO predictions (self.pred_interval_centers): {loo_preds}')
        num_chunks = math.ceil(test_contexts.shape[0] / cs)
        acts = []
        for i in range(num_chunks):
            ctxs = test_contexts[i * cs: (i+1) * cs,:] 
            # for each chunk of context, store the lower confidence bound (lcb)
            lcb = [] 
            # Calculating Lower Confidence Bound (LCB) for Each Action
            # difference between Neuralcb_cp, Neuralcb_cp does 
            # The action-specific scaling of diag_Lambda in Snippet 2 provides a more tailored calculation for each action's uncertainty, 
            # possibly leading to a more accurate representation of the uncertainty associated with each action. This could be beneficial when actions exhibit different variances.
            for a in range(self.hparams.num_actions):
                # num_actions= 2
                # a = 0 or 1
                # actions = 0 if a =0, else 1
                actions = jnp.ones(shape=(ctxs.shape[0],)) * a 
                # ************************Integration********************************
                # Use conformal predicted rewards if available, otherwise use the network's prediction
                if len(self.Ensemble_pred_interval_centers) == 0:
                    # print(f'!!!! Prediction intervals not available at current stage!!! self.Ensemble_pred_interval_centers == None')
                    f = self.nn.out(self.nn.params, ctxs, actions)  # Default to the neural network output
                    # print(f'f.shape=={f.shape}')
                    # print(f'Prediction datatype f=== {type(f)}')
                else:
                    # print(f'!!!!!! Prediction Intervals available!!!!')
                    # print(f'&&&&& len(Ensemble_pred_interval_centers)  === {len(self.Ensemble_pred_interval_centers)}&&&&&')
                    f = self.Ensemble_pred_interval_centers[i * cs: (i+1) * cs]
                    # print(f'&&&&& len(f) === {len(f)}&&&&&')
                    # print(f'PI centers datatype == {type(f)}')
                    f = jnp.array(f)
                    # print(f'PI centers datatype after converting== {type(f)}')
                # f = self.nn.out(self.nn.params, ctxs, actions) #the predicted reward for action a in the given context ctxs.
                g = self.nn.grad_out(self.nn.params, ctxs, actions) / jnp.sqrt(self.nn.m)
                # g = self.nn.grad_out_cp(self.nn.params, ctxs, actions) / jnp.sqrt(self.nn.m)
                # uncertainty of each action 
                gAg = jnp.sum(jnp.square(g) / self.diag_Lambda[a][:], axis=-1)
                # confidence
                cnf = jnp.sqrt(gAg)
                lcb_a = f.ravel() - self.hparams.beta * cnf.ravel()
                lcb.append(lcb_a.reshape(-1,1)) 
            lcb = jnp.hstack(lcb)
            # lcb_a is used to decide which action to take by selecting the action with the highest LCB.
            acts.append(jnp.argmax(lcb, axis=1)) 
        sampled_test_actions = jnp.hstack(acts)
        

        # computing preidcition intervals
        X_train = self.data.contexts
        # selected actions for training
        actions = self.data.actions.ravel().astype(int)
        # rewards for training
        Y_train = self.data.rewards[np.arange(len(actions)), actions].reshape(-1, 1)
        X_predict = test_contexts
        test_actions = sampled_test_actions.ravel().astype(int)
        Y_predict = opt_vals
        filename = self.res_dir
        nn_model = self.nn
       
        self.prediction_interval_model = prediction_interval(
                        nn_model,  
                        X_train, 
                        X_predict, 
                        Y_train, 
                        Y_predict, 
                        actions, 
                        test_actions,
                        filename,
                        self.name,
                        self.B)
        self.Ensemble_pred_interval_centers = self.prediction_interval_model.fit_bootstrap_models_online(B=self.B, miss_test_idx=[])
        # print(f'self.Ensemble_prediction_interval_centers:{self.Ensemble_pred_interval_centers}')
        PI_dfs, results = self.prediction_interval_model.run_experiments(alpha=0.05, stride=8,methods=['Ensemble'])       
        
        # return jnp.hstack(acts)
        return sampled_test_actions
    
 
    def update_buffer(self, c, a, r): 
        print(f'        !!!!!!!!!! Updating buffer !!!!!!!!!!')
        # print(f'!!!!!!!!!!data shapes before update_buffer() !!!!!!!!!!')
        # print(f'c.shape = {c.shape}')
        # print(f'a.shape = {a.shape}')
        # print(f'r.shape = {r.shape}')
        # c.shape = (1, 13)
        # a.shape = (1,)
        # r.shape = (1, 1)
        # we add one data entry at each round where there is a new hour of EHR coming 
        self.data.add(c, a, r)

    def update(self, contexts, actions, rewards):

        # training for updating the nn parameters
        self.nn.train(self.data, self.hparams.num_steps)
        u = self.nn.grad_out(self.nn.params, contexts, actions) / jnp.sqrt(self.nn.m) #neuralbanditV2
        # u  = self.nn.grad_out(self.nn.params, contexts  ) / jnp.sqrt(self.nn.m)# neuralbandit
        for i in range(contexts.shape[0]):
            self.diag_Lambda[actions[i]] = self.diag_Lambda[actions[i]] + jnp.square(u[i,:])

    def monitor(self, contexts=None, actions=None, rewards=None):
        print(f'$$$$$$$$$$$ Monitoring ApproxNeuraLCBV2 .......')
        # The way jnp.hstack is called might be causing the issue. 
        # When you use a generator expression with jnp.hstack, 
        # JAX might not handle it as expected because it could require an explicit materialization of the sequence. 
        # Try converting the generator to a list before passing it to jnp.hstack:
        # norm = jnp.hstack([jnp.ravel(param) for param in jax.tree_leaves(self.nn.params)])
        # norm = jnp.hstack((jnp.ravel(param) for param in jax.tree_leaves(self.nn.params)))
        norm = jnp.hstack([jnp.ravel(param) for param in jax.tree_leaves(self.nn.params)])
        # a flattened version of all the model parameters (weights and biases of the neural network).
        # print(f'Parameter norm: {jnp.mean(jnp.square(norm))}')
        if len(self.Ensemble_pred_interval_centers) == 0:
            preds = self.nn.out(self.nn.params, contexts, actions)    
        else:
            preds = self.Ensemble_pred_interval_centers 
        # print(f"preds.shape:{preds.shape}")
        # print(f'Predictions: {preds}')


        '''
        Both monitor() and update() compute some common quantities such as:

        f: The predicted rewards for different actions (output of the neural network).
        g: The gradient of the predicted rewards with respect to the model parameters.
        gAg: The quadratic form of the gradients and the confidence matrix, used to calculate the confidence bounds.
        However, the difference lies in how these quantities are used:

        In monitor(), these are inspected and printed for debugging.
        In update(), these values are used to compute gradients and update the model parameters.
        '''

        cnfs = []
        for a in range(self.hparams.num_actions):
            # ??? what is this actions_tmp???
            # each element is set to integer 'a', length equals to the number of contexts
            # generate a temporary action array where the chosen action a is applied uniformly across all sample in the batch
            actions_tmp = jnp.ones(shape=(contexts.shape[0],)) * a 

            #f: predicted rewards
            # seems that 'f' is not used
            # f = self.nn.out(self.nn.params, contexts, actions_tmp) # (num_samples, 1)

            g = self.nn.grad_out(self.nn.params, contexts, actions_tmp) / jnp.sqrt(self.nn.m) # (num_samples, p)
            # self.diag_Lambda[a][:] is the confidence parameters
            # this operation effectively scales the gradient by the inverse of the confidence parameters
            # providing a measure of the uncertainty of variability in the model's predictions for action a across all contexts
            gAg = jnp.sum( jnp.square(g) / self.diag_Lambda[a][:], axis=-1)
            cnf = jnp.sqrt(gAg) # (num_samples,)
            # print(f'Confidence bounds: {cnf}')
            cnfs.append(cnf) 
        cnf = jnp.hstack(cnfs) 
        cost = self.nn.loss(self.nn.params, contexts, actions, rewards )
        # if len(self.Ensemble_pred_interval_centers) > 0:
        #     print(f'length of LOO Predictions (centers): {len(self.Ensemble_pred_interval_centers)}')
        # else:
        #     print('No LOO predictions available at this point.')

        print(f'~~~~~~~~~~~~~~~~~~~========================~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')

        a = int(actions.ravel()[0])
        if self.hparams.debug_mode == 'simple':
            print('     r: {} | a: {} | f: {} | cnf: {} | loss: {} | param_mean: {}'.format(rewards.ravel()[0], a, \
                preds.ravel()[0], \
                cnf.ravel()[a], cost, jnp.mean(jnp.square(norm))))
        else:
            print('     r: {} | a: {} | f: {} | cnf: {} | loss: {} | param_mean: {}'.format(rewards.ravel()[0], \
                a, preds.ravel(), \
                cnf.ravel(), cost, jnp.mean(jnp.square(norm))))
        print(f'~~~~~~~~~~~~~~~~~~~========================~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')






     
