"""Define neural offline bandit algorithms. """

import math
import jax 
import jax.numpy as jnp
import optax
import numpy as np 
from joblib import Parallel, delayed
from core.bandit_algorithm import BanditAlgorithm 
from core.bandit_dataset import BanditDataset
from core.utils import inv_sherman_morrison, inv_sherman_morrison_single_sample, vectorize_tree
from algorithms.neural_bandit_model import NeuralBanditModel, NeuralBanditModelV2
import sys
# import utils_Sepsysolcp as util

from core.PI4nnlcb import prediction_interval
import cp_funs.utils_cp as utils_cp
 
# ======================================================================
# ========================================================================
# the ApproxNeuraLCBV2 with conformal prediction
class ApproxNeuraLCBV2_cp(BanditAlgorithm):

    def __init__(self, hparams, update_freq=1, name='ApproxNeuraLCBV2'):
        self.name = name 
        self.hparams = hparams 
        self.update_freq = update_freq
        # learning rate 1e-3
        opt = optax.adam(hparams.lr)
        self.nn = NeuralBanditModelV2(opt, hparams, '{}-net'.format(name))
        # data buffer for incoming data, update each round when we have a new (c,a, r)
        self.data = BanditDataset(hparams.context_dim, hparams.num_actions, hparams.buffer_s, '{}-data'.format(name))
        self.diag_Lambda = [jnp.ones(self.nn.num_params) * hparams.lambd0 for _ in range(hparams.num_actions)]
        self.pred_interval_centers = []
        self.prediction_interval_model = None

    def reset(self, seed): 
        self.diag_Lambda = [jnp.ones(self.nn.num_params) * self.hparams.lambd0 for _ in range(self.hparams.num_actions)]
        self.nn.reset(seed) 
        self.data.reset()
        self.prediction_interval_model = None
    # line 5 in NeuraLCB_Bmode
    def sample_action(self, contexts):
        # flags.DEFINE_integer('chunk_size', 500, 'Chunk size')
        # flags.DEFINE_integer('batch_size', 32, 'Batch size')
        cs = self.hparams.chunk_size

        num_chunks = math.ceil(contexts.shape[0] / cs)
        acts = []
        for i in range(num_chunks):
            ctxs = contexts[i * cs: (i+1) * cs,:] 
            # for each chunk of context, store the lower confidence bound (lcb)
            lcb = [] 
            for a in range(self.hparams.num_actions):
                # num_actions= 2
                # a = 0 or 1
                # actions = 0 if a =0, else 1
                actions = jnp.ones(shape=(ctxs.shape[0],)) * a 
                f = self.nn.out(self.nn.params, ctxs, actions)
                # g = self.nn.grad_out(self.nn.params, ctxs, actions) / jnp.sqrt(self.nn.m)
                g = self.nn.grad_out_cp(self.nn.params, ctxs, actions) / jnp.sqrt(self.nn.m)
                # uncertainty of each action 
                gAg = jnp.sum(jnp.square(g) / self.diag_Lambda[a][:], axis=-1)
                # confidence
                cnf = jnp.sqrt(gAg)
                lcb_a = f.ravel() - self.hparams.beta * cnf.ravel()
                lcb.append(lcb_a.reshape(-1,1)) 
            lcb = jnp.hstack(lcb)
            acts.append(jnp.argmax(lcb, axis=1)) 
        return jnp.hstack(acts)
    
 
    def update_buffer(self, contexts, actions, rewards): 
        self.data.add(contexts, actions, rewards)
    
    def update(self, contexts, actions, rewards):
        """Update the network parameters and the confidence parameter.
        
        Args:
            contexts: An array of d-dimensional contexts
            actions: An array of integers in [0, K-1] representing the chosen action 
            rewards: An array of real numbers representing the reward for (context, action)
        
        """

        # Should run self.update_buffer before self.update to update the model in the latest data. 
        loo_preds = self.pred_interval_centers
        print(f'loo_preds (self.pred_interval_centers): {loo_preds}')
        self.nn.train(self.data, self.hparams.num_steps, loo_preds)
        u = self.nn.grad_out_cp(self.nn.params, contexts, actions) / jnp.sqrt(self.nn.m)
        for i in range(contexts.shape[0]):
            self.diag_Lambda[actions[i]] = self.diag_Lambda[actions[i]] + jnp.square(u[i,:])

 

    def monitor_loo(self, contexts=None, actions=None, rewards=None):
        print(f'running monitor() of algo ApproxNeuraLCBV2 .......')
        ## debug: 
        # print(f'param.shape:')
        # print([param.shape for param in jax.tree_leaves(self.nn.params)])
        # print(f'type(self.nn.params): {self.nn.params}')
        # for param in jax.tree_leaves(self.nn.params):
        #     print(f'current param: {param}')
        #     print(f'param.shape: {param.shape}')
        #     new_param = jnp.ravel(param)
        #     print(f'new_param.shape: {new_param.shape}')
        # sys.exit()

        '''
        The way jnp.hstack is called might be causing the issue. 
        When you use a generator expression with jnp.hstack, 
        JAX might not handle it as expected because it could require an explicit materialization of the sequence. 
        Try converting the generator to a list before passing it to jnp.hstack:
        norm = jnp.hstack([jnp.ravel(param) for param in jax.tree_leaves(self.nn.params)])
        '''
        # norm = jnp.hstack((jnp.ravel(param) for param in jax.tree_leaves(self.nn.params)))
        norm = jnp.hstack([jnp.ravel(param) for param in jax.tree_leaves(self.nn.params)])
        preds = self.nn.out(self.nn.params, contexts, actions)


        cnfs = []
        for a in range(self.hparams.num_actions):
            # ??? what is this actions_tmp???
            # each element is set to integer 'a', length equals to the number of contexts
            # generate a temporary action array where the chosen action a is applied uniformly across all sample in the batch
            actions_tmp = jnp.ones(shape=(contexts.shape[0],)) * a 
            f = self.nn.out(self.nn.params, contexts, actions_tmp) # (num_samples, 1)
            # g = self.nn.grad_out(self.nn.params, contexts, actions_tmp) / jnp.sqrt(self.nn.m) # (num_samples, p)
            g = self.nn.grad_out_cp(self.nn.params, contexts, actions_tmp) / jnp.sqrt(self.nn.m) # (num_samples, p)

 
            gAg = jnp.sum( jnp.square(g) / self.diag_Lambda[a][:], axis=-1)
            cnf = jnp.sqrt(gAg) # (num_samples,)

            cnfs.append(cnf) 
        cnf = jnp.hstack(cnfs) 
        loo_preds = self.pred_interval_centers
        cost = self.nn.loss(self.nn.params, contexts, actions, rewards, loo_preds)
        a = int(actions.ravel()[0])
        if self.hparams.debug_mode == 'simple':
            print('     r: {} | a: {} | f: {} | cnf: {} | loss: {} | param_mean: {}'.format(rewards.ravel()[0], a, \
                preds.ravel()[0], \
                cnf.ravel()[a], cost, jnp.mean(jnp.square(norm))))
        else:
            print('     r: {} | a: {} | f: {} | cnf: {} | loss: {} | param_mean: {}'.format(rewards.ravel()[0], \
                a, preds.ravel(), \
                cnf.ravel(), cost, jnp.mean(jnp.square(norm))))
    '''
    def monitor(self, contexts=None, actions=None, rewards=None):
        print(f'running monitor() of algo ApproxNeuraLCBV2 .......')
        loo_preds = self.get_loo_params(contexts, actions, rewards)
        
        # The way jnp.hstack is called might be causing the issue. 
        # When you use a generator expression with jnp.hstack, 
        # JAX might not handle it as expected because it could require an explicit materialization of the sequence. 
        # Try converting the generator to a list before passing it to jnp.hstack:
        # norm = jnp.hstack([jnp.ravel(param) for param in jax.tree_leaves(self.nn.params)])
        # norm = jnp.hstack((jnp.ravel(param) for param in jax.tree_leaves(self.nn.params)))
        norm = jnp.hstack([jnp.ravel(param) for param in jax.tree_leaves(self.nn.params)])

        preds = self.nn.out(self.nn.params, contexts, actions)
        # print(f"preds.shape:{preds.shape}")

        cnfs = []
        for a in range(self.hparams.num_actions):
            # ??? what is this actions_tmp???
            # each element is set to integer 'a', length equals to the number of contexts
            # generate a temporary action array where the chosen action a is applied uniformly across all sample in the batch
            actions_tmp = jnp.ones(shape=(contexts.shape[0],)) * a 
            #f: predicted rewards
            # seems that 'f' is not used
            f = self.nn.out(self.nn.params, contexts, actions_tmp) # (num_samples, 1)
            g = self.nn.grad_out(self.nn.params, contexts, actions_tmp) / jnp.sqrt(self.nn.m) # (num_samples, p)
            # self.diag_Lambda[a][:] is the confidence parameters
            # this operation effectively scales the gradient by the inverse of the confidence parameters
            # providing a measure of the uncertainty of variability in the model's predictions for action a across all contexts
            gAg = jnp.sum( jnp.square(g) / self.diag_Lambda[a][:], axis=-1)
            cnf = jnp.sqrt(gAg) # (num_samples,)

            cnfs.append(cnf) 
        cnf = jnp.hstack(cnfs) 
        loo_preds = self.pred_interval_centers
        cost = self.nn.loss(self.nn.params, contexts, actions, rewards, loo_preds)
        a = int(actions.ravel()[0])
        if self.hparams.debug_mode == 'simple':
            print('     r: {} | a: {} | f: {} | cnf: {} | loss: {} | param_mean: {}'.format(rewards.ravel()[0], a, \
                preds.ravel()[0], \
                cnf.ravel()[a], cost, jnp.mean(jnp.square(norm))))
        else:
            print('     r: {} | a: {} | f: {} | cnf: {} | loss: {} | param_mean: {}'.format(rewards.ravel()[0], \
                a, preds.ravel(), \
                cnf.ravel(), cost, jnp.mean(jnp.square(norm))))
    '''

            