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
import utils_Sepsysolcp as util
from core.PI4nnlcb import prediction_interval


class ExactNeuraLCBV2(BanditAlgorithm):
    """NeuraLCB using exact confidence matrix and NeuralBanditModelV2. """
    def __init__(self, hparams, update_freq=1, name='ExactNeuraLCBV2'):
        self.name = name 
        self.hparams = hparams 
        self.update_freq = update_freq
        opt = optax.adam(hparams.lr)
        # opt = optax.adam(self.hparams.lr)
        self.nn = NeuralBanditModelV2(opt, hparams, '{}-net'.format(name))
        # self.nn = NeuralBanditModelV2(opt, self.hparams, '{}-net'.format(self.name))
      
        self.data = BanditDataset(hparams.context_dim, hparams.num_actions, hparams.buffer_s, '{}-data'.format(name))
        # self.nn.num_params: the number of network parameters (i.e., p)
        self.Lambda_inv = jnp.array(
            [
                jnp.eye(self.nn.num_params)/hparams.lambd0 for _ in range(hparams.num_actions)
            ]
        ) # (num_actions, p, p)

    def reset(self, seed): 
        self.Lambda_inv = jnp.array(
            [
                jnp.eye(self.nn.num_params)/ self.hparams.lambd0 for _ in range(self.hparams.num_actions)
            ]
        ) # (num_actions, p, p)

        self.nn.reset(seed) 
        # this will reset actions, contexts, rewards to None
        self.data.reset()

    def sample_action(self, contexts):
        """
        Args:
            context: (None, self.hparams.context_dim)
        """
        cs = self.hparams.chunk_size
        num_chunks = math.ceil(contexts.shape[0] / cs)
        acts = []
        for i in range(num_chunks):
            ctxs = contexts[i * cs: (i+1) * cs,:] 
            lcb = []
            for a in range(self.hparams.num_actions):
                actions = jnp.ones(shape=(ctxs.shape[0],)) * a 
                # this is the predicted rewards 
                f = self.nn.out(self.nn.params, ctxs, actions) # (num_samples, 1)
                # g = self.nn.grad_out(self.nn.params, convoluted_contexts) / jnp.sqrt(self.nn.m) # (num_samples, p)
                g = self.nn.grad_out(self.nn.params, ctxs, actions) / jnp.sqrt(self.nn.m)
                gA = g @ self.Lambda_inv[a,:,:] # (num_samples, p)
                
                gAg = jnp.sum(jnp.multiply(gA, g), axis=-1) # (num_samples, )
                cnf = jnp.sqrt(gAg) # (num_samples,)

                lcb_a = f.ravel() - self.hparams.beta * cnf.ravel()  # (num_samples,)
                lcb.append(lcb_a.reshape(-1,1)) 
            lcb = jnp.hstack(lcb) 
            acts.append( jnp.argmax(lcb, axis=1)) 
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

        # self.data.add(contexts, actions, rewards)
        self.nn.train(self.data, self.hparams.num_steps)

        # Update confidence parameter over all samples in the batch
        # convoluted_contexts = self.nn.action_convolution(contexts, actions)  
        # u = self.nn.grad_out(self.nn.params, convoluted_contexts) / jnp.sqrt(self.nn.m)  # (num_samples, p)
        u = self.nn.grad_out(self.nn.params, contexts, actions) / jnp.sqrt(self.nn.m)
        for i in range(contexts.shape[0]):
            jax.ops.index_update(self.Lambda_inv, actions[i], \
                inv_sherman_morrison_single_sample(u[i,:], self.Lambda_inv[actions[i],:,:]))

    def monitor(self, contexts=None, actions=None, rewards=None):
        norm = jnp.hstack(( jnp.ravel(param) for param in jax.tree_leaves(self.nn.params)))

        preds = self.nn.out(self.nn.params, contexts, actions) # (num_samples,)

        cnfs = []
        for a in range(self.hparams.num_actions):
            actions_tmp = jnp.ones(shape=(contexts.shape[0],)) * a 

            f = self.nn.out(self.nn.params, contexts, actions_tmp) # (num_samples, 1)
            g = self.nn.grad_out(self.nn.params, contexts, actions_tmp) / jnp.sqrt(self.nn.m) # (num_samples, p)
            gA = g @ self.Lambda_inv[a,:,:] # (num_samples, p)
            
            gAg = jnp.sum(jnp.multiply(gA, g), axis=-1) # (num_samples, )
            cnf = jnp.sqrt(gAg) # (num_samples,)

            cnfs.append(cnf) 
        cnf = jnp.hstack(cnfs) 

        cost = self.nn.loss(self.nn.params, contexts, actions, rewards)
        a = int(actions.ravel()[0])
        if self.hparams.debug_mode == 'simple':
            print('     r: {} | a: {} | f: {} | cnf: {} | loss: {} | param_mean: {}'.format(rewards.ravel()[0], a, \
                preds.ravel()[0], \
                cnf.ravel()[a], cost, jnp.mean(jnp.square(norm))))
        else:
            print('     r: {} | a: {} | f: {} | cnf: {} | loss: {} | param_mean: {}'.format(rewards.ravel()[0], \
                a, preds.ravel(), \
                cnf.ravel(), cost, jnp.mean(jnp.square(norm))))

## 
'''
an example of hparams: 
    hparams = edict({
        'layer_sizes': [100,100], 
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

'''

# the ApproxNeuraLCBV2 for conformal prediction
class ApproxNeuraLCBV2(BanditAlgorithm):
    def __init__(self, hparams, update_freq=1, name='ApproxNeuraLCBV2'):
        self.name = name 
        self.hparams = hparams 
        self.update_freq = update_freq
        opt = optax.adam(hparams.lr)
        self.nn = NeuralBanditModelV2(opt, hparams, '{}-net'.format(name))
        self.data = BanditDataset(hparams.context_dim, hparams.num_actions, hparams.buffer_s, '{}-data'.format(name))
        self.diag_Lambda = [jnp.ones(self.nn.num_params) * hparams.lambd0 for _ in range(hparams.num_actions)]

    def reset(self, seed): 
        self.diag_Lambda = [jnp.ones(self.nn.num_params) * self.hparams.lambd0 for _ in range(self.hparams.num_actions)]
        self.nn.reset(seed) 
        self.data.reset()
                             
    def sample_action(self, contexts):
        cs = self.hparams.chunk_size
        num_chunks = math.ceil(contexts.shape[0] / cs)
        acts = []
        for i in range(num_chunks):
            ctxs = contexts[i * cs: (i+1) * cs,:] 
            lcb = []
            for a in range(self.hparams.num_actions):
                actions = jnp.ones(shape=(ctxs.shape[0],)) * a 
                f = self.nn.out(self.nn.params, ctxs, actions)
                g = self.nn.grad_out(self.nn.params, ctxs, actions) / jnp.sqrt(self.nn.m)
                gAg = jnp.sum(jnp.square(g) / self.diag_Lambda[a][:], axis=-1)
                cnf = jnp.sqrt(gAg)
                lcb_a = f.ravel() - self.hparams.beta * cnf.ravel()
                lcb.append(lcb_a.reshape(-1,1)) 
            lcb = jnp.hstack(lcb)
            acts.append(jnp.argmax(lcb, axis=1)) 
        return jnp.hstack(acts)

    def update_buffer(self, contexts, actions, rewards): 
        self.data.add(contexts, actions, rewards)
    
    def update(self, contexts, actions, rewards):
        self.nn.train(self.data, self.hparams.num_steps)
        u = self.nn.grad_out(self.nn.params, contexts, actions) / jnp.sqrt(self.nn.m)
        for i in range(contexts.shape[0]):
            self.diag_Lambda[actions[i]] = self.diag_Lambda[actions[i]] + jnp.square(u[i,:])

    def monitor(self, contexts=None, actions=None, rewards=None):
        print(f'running monitor() of algo ApproxNeuraLCBV2 .......')
        norm = jnp.hstack([jnp.ravel(param) for param in jax.tree_leaves(self.nn.params)])
        print(f'norm:{norm}')
        print(f'norm.shape:{norm.shape}')
        preds = self.nn.out(self.nn.params, contexts, actions)
        print(f"contexts.shape:{contexts.shape}")
        print(f"preds.shape:{preds.shape}")

        cnfs = []
        for a in range(self.hparams.num_actions):
            actions_tmp = jnp.ones(shape=(contexts.shape[0],)) * a
            f = self.nn.out(self.nn.params, contexts, actions_tmp)
            g = self.nn.grad_out(self.nn.params, contexts, actions_tmp) / jnp.sqrt(self.nn.m)
            gAg = jnp.sum(jnp.square(g) / self.diag_Lambda[a][:], axis=-1)
            cnf = jnp.sqrt(gAg)
            cnfs.append(cnf)
        cnf = jnp.hstack(cnfs)
        cost = self.nn.loss(self.nn.params, contexts, actions, rewards)
        a = int(actions.ravel()[0])
        if self.hparams.debug_mode == 'simple':
            print('     r: {} | a: {} | f: {} | cnf: {} | loss: {} | param_mean: {}'.format(rewards.ravel()[0], a, preds.ravel()[0], cnf.ravel()[a], cost, jnp.mean(jnp.square(norm))))
        else:
            print('     r: {} | a: {} | f: {} | cnf: {} | loss: {} | param_mean: {}'.format(rewards.ravel()[0], a, preds.ravel(), cnf.ravel(), cost, jnp.mean(jnp.square(norm))))

        # Extract training and prediction data from the bandit dataset
        X_train, Y_train = self.data.contexts, self.data.rewards
        X_predict, Y_predict = contexts, rewards
        
        # Initialize prediction_interval_model with precomputed predictions
        self.prediction_interval_model = prediction_interval(
            self.nn,  # NeuralBanditModelV2 instance
            X_train, X_predict, Y_train, Y_predict,
            precomputed_preds=preds.ravel()
        )
        
        PIs_df, mean_coverage = self.prediction_interval_model.run_experiments(0.05, 10, 1, 'dataset_name', 0, [], get_plots=False)
        Y_upper = PIs_df[0]['upper'].values
        Y_lower = PIs_df[0]['lower'].values
        print(f'Prediction Intervals: Lower: {Y_lower}, Upper: {Y_upper}')
        print(f'Coverage: {mean_coverage}')


        # def process(i):
        #     ctxs = contexts[i * cs: (i+1) * cs,:] 
        #     lcb = []
        #     for a in range(self.hparams.num_actions):
        #         actions = jnp.ones(shape=(ctxs.shape[0],)) * a 

        #         f = self.nn.out(self.nn.params, ctxs, actions) # (num_samples, 1)
        #         # g = self.nn.grad_out(self.nn.params, convoluted_contexts) / jnp.sqrt(self.nn.m) # (num_samples, p)
        #         g = self.nn.grad_out(self.nn.params, ctxs, actions) / jnp.sqrt(self.nn.m)
        #         gAg = jnp.sum( jnp.square(g) / self.diag_Lambda[a][:], axis=-1) # (None, p) -> (None,)

        #         cnf = jnp.sqrt(gAg) # (num_samples,)
        #         lcb_a = f.ravel() - self.hparams.beta * cnf.ravel()  # (num_samples,)
        #         lcb.append(lcb_a.reshape(-1,1)) 
        #     lcb = jnp.hstack(lcb) 
        #     # print(lcb)
        #     return jnp.argmax(lcb, axis=1)
    
        # acts = Parallel(n_jobs=50,prefer="threads")(delayed(process)(i) for i in range(num_chunks))
        # return jnp.hstack(acts)
    

    # see the definition in BanditDataset\
    # update contexts, actions and reward
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
        self.nn.train(self.data, self.hparams.num_steps)

        # Update confidence parameter over all samples in the batch
        # convoluted_contexts = self.nn.action_convolution(contexts, actions)  
        # u = self.nn.grad_out(self.nn.params, convoluted_contexts) / jnp.sqrt(self.nn.m)  # (num_samples, p)
        # grad_out(params, contexts)
        u = self.nn.grad_out(self.nn.params, contexts, actions) / jnp.sqrt(self.nn.m)
        for i in range(contexts.shape[0]):
            # jax.ops.index_update(self.diag_Lambda, actions[i], \
            #     jnp.square(u[i,:]) + self.diag_Lambda[actions[i],:])  
            # mapped to pseudo code line 6 in the paper
            #'self.diag_Lambda' is initialized as a list of diagonal matrices 
            # one for each action
            '''
            self.diag_Lambda stores these parameters for each possible action, 
            actions[i] specifies which action's parameters should be updated based on the current observation. 
            This allows the algorithm to update its confidence levels uniquely for each action based on the data observed, 
            enabling more nuanced decision-making in the contextual bandit setting.
            '''
            self.diag_Lambda[actions[i]] = self.diag_Lambda[actions[i]] + jnp.square(u[i,:])
            # print(f'actions.shape:{actions.shape}')
            # print(f'actions[i]:{actions[i]}')
            # print(f'type(self.diag_Lambda):{type(self.diag_Lambda)}')
            # print(f'len(self.diag_Lambda):{len(self.diag_Lambda)}')
    
    def calculate_loo_residuals(self):
        n = len(self.X_train)

    def monitor(self, contexts=None, actions=None, rewards=None):
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
        print(f'norm:{norm}')
        print(f'norm.shape:{norm.shape}')

        # sys.exit()
        # print(f'norm of ApproxNeuraLCBV2: {norm}')
        # for key in self.nn.params.keys():
        #     print(f'key:{key}')
        #     print(f'self.nn.params[{key}]:{self.nn.params[key]}')
        # # show the params after applying jax.tree_leaves:
        # for param in jax.tree_leaves(self.nn.params):
        #     print(f'param.shape in tree_leaves(self.nn.params):{param.shape}')
        #     print(f'param in tree_leaves(self.nn.params):{param}')
        # sys.exit()


        # this is the predictions from the neural networks????
        # (num_samples,)

        print(f"contexts.shape:{contexts.shape}")
        preds = self.nn.out(self.nn.params, contexts, actions)
        print(f"preds.shape:{preds.shape}")


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

        cost = self.nn.loss(self.nn.params, contexts, actions, rewards)
        a = int(actions.ravel()[0])
        if self.hparams.debug_mode == 'simple':
            print('     r: {} | a: {} | f: {} | cnf: {} | loss: {} | param_mean: {}'.format(rewards.ravel()[0], a, \
                preds.ravel()[0], \
                cnf.ravel()[a], cost, jnp.mean(jnp.square(norm))))
        else:
            print('     r: {} | a: {} | f: {} | cnf: {} | loss: {} | param_mean: {}'.format(rewards.ravel()[0], \
                a, preds.ravel(), \
                cnf.ravel(), cost, jnp.mean(jnp.square(norm))))
            
        ## calculating conformal prediction intervals
        # Extract training and prediction data from the bandit dataset
        X_train, Y_train = self.data.contexts, self.data.rewards
        X_predict, Y_predict = contexts, rewards
        
        # Initialize prediction_interval_model with precomputed predictions
        self.prediction_interval_model = prediction_interval(
            self.nn,  # NeuralBanditModelV2 instance
            X_train, X_predict, Y_train, Y_predict
            # precomputed_preds=preds.ravel()
        )
        
        PIs_df, mean_coverage = self.prediction_interval_model.run_experiments(0.05, 10, 1, 'dataset_name', 0, [], get_plots=False)
        Y_upper = PIs_df[0]['upper'].values
        Y_lower = PIs_df[0]['lower'].values
        print(f'Prediction Intervals: Lower: {Y_lower}, Upper: {Y_upper}')
        print(f'Coverage: {mean_coverage}')

class NeuralGreedyV2(BanditAlgorithm):
    def __init__(self, hparams, update_freq=1, name='NeuralGreedyV2'):
        self.name = name 
        self.hparams = hparams 
        self.update_freq = update_freq 
        opt = optax.adam(hparams.lr)
        self.nn = NeuralBanditModelV2(opt, hparams, '{}-net'.format(name))
        self.data = BanditDataset(hparams.context_dim, hparams.num_actions, hparams.buffer_s, '{}-data'.format(name))

    def reset(self, seed): 
        self.nn.reset(seed) 
        self.data.reset()

    def sample_action(self, contexts):
        preds = []
        for a in range(self.hparams.num_actions):
            actions = jnp.ones(shape=(contexts.shape[0],)) * a 
            f = self.nn.out(self.nn.params, contexts, actions) # (num_samples, 1)
            preds.append(f) 
        preds = jnp.hstack(preds) 
        return jnp.argmax(preds, axis=1)

    def update_buffer(self, contexts, actions, rewards): 
        self.data.add(contexts, actions, rewards)

    def update(self, contexts, actions, rewards):
        """Update the network parameters and the confidence parameter.
        
        Args:
            context: An array of d-dimensional contexts
            action: An array of integers in [0, K-1] representing the chosen action 
            reward: An array of real numbers representing the reward for (context, action)
        
        """

        # self.data.add(contexts, actions, rewards)
        self.nn.train(self.data, self.hparams.num_steps)

    def monitor(self, contexts=None, actions=None, rewards=None):
        norm = jnp.hstack(( jnp.ravel(param) for param in jax.tree_leaves(self.nn.params)))

        convoluted_contexts = self.nn.action_convolution(contexts, actions)

        preds = self.nn.out(self.nn.params, contexts, actions) # (num_samples,)

        preds = []
        for a in range(self.hparams.num_actions):
            actions_tmp = jnp.ones(shape=(contexts.shape[0],)) * a 
            f = self.nn.out(self.nn.params, contexts, actions_tmp) # (num_samples, 1)
            preds.append(f) 
        preds = jnp.hstack(preds) 

        cost = self.nn.loss(self.nn.params, contexts, actions, rewards)

        a = int(actions.ravel()[0])
        if self.hparams.debug_mode == 'simple':
            print('     r: {} | a: {} | f: {} | loss: {} | param_mean: {}'.format(rewards.ravel()[0], \
            a, preds.ravel()[a], \
            cost, jnp.mean(jnp.square(norm)))
            )
        else:
            print('     r: {} | a: {} | f: {} | loss: {} | param_mean: {}'.format(rewards.ravel()[0], \
                a, preds.ravel(), \
                cost, jnp.mean(jnp.square(norm)))
                )
#===================================================================================================

class NeuraLCB(BanditAlgorithm):
    """NeuraLCB using diag approximation for confidence matrix. """
    def __init__(self, hparams, name='NeuraLCB'):
        self.name = name 
        self.hparams = hparams 
        opt = optax.adam(hparams.lr)
        self.nn = NeuralBanditModel(opt, hparams, 'nn')
        self.data = BanditDataset(hparams.context_dim, hparams.num_actions, hparams.buffer_s, 'bandit_data')

        self.diag_Lambda = jnp.array(
                [hparams.lambd0 * jnp.ones(self.nn.num_params) for _ in range(self.hparams.num_actions) ]
            ) # (num_actions, p)

    def sample_action(self, contexts):
        """
        Args:
            contexts: (None, self.hparams.context_dim)
        """
        n = contexts.shape[0] 
        
        if n <= self.hparams.max_test_batch:
            f = self.nn.out(self.nn.params, contexts) # (num_samples, num_actions)
            g = self.nn.grad_out(self.nn.params, contexts) # (num_actions, num_samples, p)

            cnf = jnp.sqrt( jnp.sum(jnp.square(g) / self.diag_Lambda.reshape(self.hparams.num_actions,1,-1), axis=-1) ) / jnp.sqrt(self.nn.m)
            lcb = f - self.hparams.beta * cnf.T   # (num_samples, num_actions)
            return jnp.argmax(lcb, axis=1)
        else: # Break contexts in batches if it is large.
            inv = int(n / self.hparams.max_test_batch)
            acts = []
            for i in range(inv):
                c = contexts[i*self.hparams.max_test_batch:self.hparams.max_test_batch*(i+1),:]
                f = self.nn.out(self.nn.params, c) # (num_samples, num_actions)
                g = self.nn.grad_out(self.nn.params, c) # (num_actions, num_samples, p)

                cnf = jnp.sqrt( jnp.sum(jnp.square(g) / self.diag_Lambda.reshape(self.hparams.num_actions,1,-1), axis=-1) ) / jnp.sqrt(self.nn.m)
                lcb = f - self.hparams.beta * cnf.T   # (num_samples, num_actions)
                acts.append(jnp.argmax(lcb, axis=1).ravel())
            return jnp.array(acts)
            

    def update(self, contexts, actions, rewards):
        """Update the network parameters and the confidence parameter.
        
        Args:
            contexts: An array of d-dimensional contexts
            actions: An array of integers in [0, K-1] representing the chosen action 
            rewards: An array of real numbers representing the reward for (context, action)
        
        """

        self.data.add(contexts, actions, rewards)
        self.nn.train(self.data, self.hparams.num_steps)

        # Update confidence parameter over all samples in the batch
        g = self.nn.grad_out(self.nn.params, contexts)  # (num_actions, num_samples, p)
        g = jnp.square(g) / self.nn.m 
        for i in range(g.shape[1]): 
            self.diag_Lambda += g[:,i,:]

    def monitor(self, contexts=None, actions=None, rewards=None):
        params = jnp.hstack(( jnp.ravel(param) for param in jax.tree_leaves(self.nn.params)))

        f = self.nn.out(self.nn.params, contexts) # (num_samples, num_actions)
        g = self.nn.grad_out(self.nn.params, contexts) # (num_actions, num_samples, p)

        cnf = jnp.sqrt( jnp.sum(jnp.square(g) / self.diag_Lambda.reshape(self.hparams.num_actions,1,-1), axis=-1) ) / jnp.sqrt(self.nn.m)
        
        # action and reward fed here are in vector forms, we convert them into an array of one-hot vectors
        action_hot = jax.nn.one_hot(actions.ravel(), self.hparams.num_actions) 
        reward_hot = action_hot * rewards.reshape(-1,1)

        cost = self.nn.loss(self.nn.params, contexts, action_hot, reward_hot)
        print('     r: {} | a: {} | f: {} | cnf: {} | param_mean: {}'.format(rewards.ravel()[0], actions.ravel()[0], f.ravel(), \
            cnf.ravel(), jnp.mean(jnp.square(params))))

class ExactNeuraLCB(BanditAlgorithm):
    """NeuraLCB using exact confidence matrix. """
    def __init__(self, hparams, name='ExactNeuraLCB'):
        self.name = name 
        self.hparams = hparams 
        opt = optax.adam(hparams.lr)
        self.nn = NeuralBanditModel(opt, hparams, 'nn')
        self.data = BanditDataset(hparams.context_dim, hparams.num_actions, hparams.buffer_s, 'bandit_data')

        # self.diag_Lambda = jnp.array(
        #         [hparams.lambd0 * jnp.ones(self.nn.num_params) for _ in range(self.hparams.num_actions) ]
        #     ) # (num_actions, p)

        self.Lambda_inv = jnp.array(
            [
                np.eye(self.nn.num_params)/hparams.lambd0 for _ in hparams.num_actions
            ]
        ) # (num_actions, p, p)

    def sample_action(self, contexts):
        """
        Args:
            context: (None, self.hparams.context_dim)
        """
        n = contexts.shape[0] 
        
        if n <= self.hparams.max_test_batch:
            f = self.nn.out(self.nn.params, context) # (num_samples, num_actions)
            g = self.nn.grad_out(self.nn.params, context) / jnp.sqrt(self.nn.m) # (num_actions, num_samples, p)
            gA = jnp.sum(jnp.multiply(g[:,:,None,:], self.Lambda_inv[:, None, :,:]), axis=-1) # (num_actions, num_samples, p)
            gAg = jnp.sum(jnp.multiply(gA, g), axis=-1) # (num_actions, num_samples)
            cnf = jnp.sqrt(gAg).T # (num_samples, num_actions)

            lcb = f - self.hparams.beta * cnf   # (num_samples, num_actions)
            return jnp.argmax(lcb, axis=1)
        else: # Break contexts in batches if it is large.
            inv = int(n / self.hparams.max_test_batch)
            acts = []
            for i in range(inv):
                c = context[i*self.hparams.max_test_batch:self.hparams.max_test_batch*(i+1),:]
                f = self.nn.out(self.nn.params, c) # (num_samples, num_actions)
                g = self.nn.grad_out(self.nn.params, c) # (num_actions, num_samples, p)

                gA = jnp.sum(jnp.multiply(g[:,:,None,:], self.Lambda_inv[:, None, :,:]), axis=-1) # (num_actions, num_samples, p)
                gAg = jnp.sum(jnp.multiply(gA, g), axis=-1) # (num_actions, num_samples)
                cnf = jnp.sqrt(gAg).T # (num_samples, num_actions)

                lcb = f - self.hparams.beta * cnf   # (num_samples, num_actions)
                acts.append(jnp.argmax(lcb, axis=1).ravel())
            return jnp.array(acts)
            

    def update(self, contexts, actions, rewards):
        """Update the network parameters and the confidence parameter.
        
        Args:
            contexts: An array of d-dimensional contexts, (None, context_dim)
            actions: An array of integers in [0, K-1] representing the chosen action, (None,)
            rewards: An array of real numbers representing the reward for (context, action), (None, )
        
        """

        self.data.add(contexts, actions, rewards)
        self.nn.train(self.data, self.hparams.num_steps)

        # Update confidence parameter over all samples in the batch
        u = self.nn.grad_out(self.nn.params, contexts) / jnp.sqrt(self.nn.m)  # (num_actions, num_samples, p)
        for i in range(g.shape[1]): 
            self.Lambda_inv =  inv_sherman_morrison(u[:,i,:], self.Lambda_inv)

    def monitor(self, contexts=None, actions=None, rewards=None):
        params = jnp.hstack(( jnp.ravel(param) for param in jax.tree_leaves(self.nn.params)))

        f = self.nn.out(self.nn.params, contexts) # (num_samples, num_actions)
        g = self.nn.grad_out(self.nn.params, contexts) # (num_actions, num_samples, p)

        gA = jnp.sum(jnp.multiply(g[:,:,None,:], self.Lambda_inv[:, None, :,:]), axis=-1) # (num_actions, num_samples, p)
        gAg = jnp.sum(jnp.multiply(gA, g), axis=-1) # (num_actions, num_samples)
        cnf = jnp.sqrt(gAg).T # (num_samples, num_actions)
        
        # action and reward fed here are in vector forms, we convert them into an array of one-hot vectors for computing loss
        action_hot = jax.nn.one_hot(actions.ravel(), self.hparams.num_actions) 
        reward_hot = action_hot * rewards.reshape(-1,1)

        cost = self.nn.loss(self.nn.params, contexts, action_hot, reward_hot)
        print('     r: {} | a: {} | f: {} | cnf: {} | param_mean: {}'.format(rewards.ravel()[0], actions.ravel()[0], f.ravel(), \
            cnf.ravel(), jnp.mean(jnp.square(params))))


class NeuralGreedy(BanditAlgorithm):
    def __init__(self, hparams, name='NeuralGreedy'):
        self.name = name 
        self.hparams = hparams 
        opt = optax.adam(hparams.lr)
        self.nn = NeuralBanditModel(opt, hparams, 'nn')
        self.data = BanditDataset(hparams.context_dim, hparams.num_actions, hparams.buffer_s, 'bandit_data')

    def sample_action(self, contexts):
        f = self.nn.out(self.nn.params, contexts) # (num_contexts, num_actions)
        return jnp.argmax(f, axis=1)

    def update(self, contexts, actions, rewards):
        """Update the network parameters and the confidence parameter.
        
        Args:
            context: An array of d-dimensional contexts
            action: An array of integers in [0, K-1] representing the chosen action 
            reward: An array of real numbers representing the reward for (context, action)
        
        """

        self.data.add(contexts, actions, rewards)
        self.nn.train(self.data, self.hparams.num_steps)

    def monitor(self, contexts=None, actions=None, rewards=None):
        params = jnp.hstack(( jnp.ravel(param) for param in jax.tree_leaves(self.nn.params)))
        f = self.nn.out(self.nn.params, contexts) # (num_samples, num_actions)

        # action and reward fed here are in vector forms, we convert them into an array of one-hot vectors
        action_hot = jax.nn.one_hot(actions.ravel(), self.hparams.num_actions) 
        reward_hot = action_hot * rewards.reshape(-1,1)
        cost = self.nn.loss(self.nn.params, contexts, action_hot, reward_hot)
        print('     r: {} | a: {} | f: {} | param_mean: {}'.format(rewards.ravel()[0], actions.ravel()[0], f.ravel(), jnp.mean(jnp.square(params))))


