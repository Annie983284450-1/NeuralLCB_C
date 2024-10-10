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
 
import importlib
import cp_funs.PI

importlib.reload(cp_funs.PI)
from cp_funs.PI import prediction_interval


#this is actually the algo in the paper. it should be the best-performing. theoretically I mean
class ExactNeuraLCBV2_cp(BanditAlgorithm):
    """NeuraLCB using exact confidence matrix and NeuralBanditModelV2. """
    def __init__(self, hparams,res_dir,B,  update_freq=1, name='ExactNeuraLCBV2_cp'):
        self.name = name 
        self.B = B
        self.hparams = hparams 
        self.update_freq = update_freq
        opt = optax.adam(hparams.lr)
        # opt = optax.adam(self.hparams.lr)
        self.nn = NeuralBanditModelV2(opt, hparams, '{}_nn2'.format(self.name))
       
        self.data = BanditDataset(hparams.context_dim, hparams.num_actions, hparams.buffer_s, '{}-data'.format(name))
        # self.nn.num_params: the number of network parameters (i.e., p)

        self.Lambda_inv = jnp.array(
            [
                jnp.eye(self.nn.num_params)/hparams.lambd0 for _ in range(hparams.num_actions)
            ]
        ) # (num_actions, p, p)
        self.prediction_interval_model = None
        self.res_dir  = res_dir
        self.Ensemble_pred_interval_centers = []   

 
    def reset(self, seed): 
        self.Lambda_inv = jnp.array(
            [
                jnp.eye(self.nn.num_params)/ self.hparams.lambd0 for _ in range(self.hparams.num_actions)
            ]
        ) # (num_actions, p, p)

        self.nn.reset(seed) 
        # this will reset actions, contexts, rewards to None
        self.data.reset()
        print(f'~~~~~~~!!!!!! After running algo.reset()~!!!!!!!!!!~~~~~~~~~~')
        print(f'self.data.rewards.shape:{self.data.rewards}')


    def sample_action(self, contexts,  opt_vals, opt_actions):
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
                if len(self.Ensemble_pred_interval_centers) == 0:
                    # print(f'________________{i}-th chunk________________')
                    # print(f'!!!! Prediction intervals not available at current stage!!! self.Ensemble_pred_interval_centers == None')
                    f = self.nn.out(self.nn.params, ctxs, actions)  # Default to the neural network output
                else:
                    # print(f'!!!!!! Prediction Intervals available!!!!')
                    # print(f'&&&&& len(Ensemble_pred_interval_centers)  === {len(self.Ensemble_pred_interval_centers)}&&&&&')
                    f = self.Ensemble_pred_interval_centers[i * cs: (i+1) * cs]
                    f = jnp.array(f)
                # g = self.nn.grad_out(self.nn.params, convoluted_contexts) / jnp.sqrt(self.nn.m) # (num_samples, p)
                g = self.nn.grad_out(self.nn.params, ctxs, actions) / jnp.sqrt(self.nn.m)
                gA = g @ self.Lambda_inv[a,:,:] # (num_samples, p)
                
                gAg = jnp.sum(jnp.multiply(gA, g), axis=-1) # (num_samples, )
                cnf = jnp.sqrt(gAg) # (num_samples,)

                lcb_a = f.ravel() - self.hparams.beta * cnf.ravel()  # (num_samples,)
                lcb.append(lcb_a.reshape(-1,1)) 
            lcb = jnp.hstack(lcb) 
            acts.append(jnp.argmax(lcb, axis=1)) 
        sampled_test_actions = jnp.hstack(acts)

        X_train = self.data.contexts
        # selected actions for training
        actions = self.data.actions.ravel().astype(int)
        # rewards for training
        Y_train = self.data.rewards[np.arange(len(actions)), actions].reshape(-1, 1)
        X_predict = contexts
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
        return sampled_test_actions
    def update_buffer(self, contexts, actions, rewards): 
        self.data.add(contexts, actions, rewards)
        
    def update(self, contexts, actions, rewards):
        """Update the network parameters and the confidence parameter.
        
        Args:
            contexts: An array of d-dimensional contexts
            actions: An array of integers in [0, K-1] representing the chosen action 
            rewards: An array of real numbers representing the reward for (context, action)
        
        """
        
        self.nn.train(self.data, self.hparams.num_steps)
        u = self.nn.grad_out(self.nn.params, contexts, actions) / jnp.sqrt(self.nn.m)
        for i in range(contexts.shape[0]):
            # jax.ops.index_update(self.Lambda_inv, actions[i], \
            #     inv_sherman_morrison_single_sample(u[i,:], self.Lambda_inv[actions[i],:,:]))
            self.Lambda_inv = self.Lambda_inv.at[actions[i]].set(inv_sherman_morrison_single_sample(u[i, :], self.Lambda_inv[actions[i], :, :]))



    def monitor(self, contexts=None, actions=None, rewards=None):
        # print("self.nn.params:", self.nn.params)
        # print("tree_leaves(self.nn.params):", jax.tree_leaves(self.nn.params))


        # for param in jax.tree_leaves(self.nn.params):
        #     print(f'param.shape in jax.tree_leaves(self.nn.params):')
        #     print(param.shape)
        # sys.exit()
        # norm = jnp.hstack(( jnp.ravel(param) for param in jax.tree_leaves(self.nn.params)))
        norm = jnp.hstack([jnp.ravel(param) if param.shape != (1,) else param.reshape(1,) for param in jax.tree_leaves(self.nn.params)])


        # preds = self.nn.out(self.nn.params, contexts, actions) # (num_samples,)
        
        if len(self.Ensemble_pred_interval_centers) == 0:
            preds = self.nn.out(self.nn.params, contexts, actions)  
        else:
            preds = self.Ensemble_pred_interval_centers 

        cnfs = []
        for a in range(self.hparams.num_actions):
            actions_tmp = jnp.ones(shape=(contexts.shape[0],)) * a 

            # f = self.nn.out(self.nn.params, contexts, actions_tmp) # (num_samples, 1)
            g = self.nn.grad_out(self.nn.params, contexts, actions_tmp) / jnp.sqrt(self.nn.m) # (num_samples, p)
            gA = g @ self.Lambda_inv[a,:,:] # (num_samples, p)
            
            gAg = jnp.sum(jnp.multiply(gA, g), axis=-1) # (num_samples, )
            cnf = jnp.sqrt(gAg) # (num_samples,)

            cnfs.append(cnf) 
        cnf = jnp.hstack(cnfs) 

        cost = self.nn.loss(self.nn.params, contexts, actions, rewards)

        if len(self.Ensemble_pred_interval_centers) > 0:
            print(f'length of LOO Predictions (centers): {len(self.Ensemble_pred_interval_centers)}')
        else:
            print('No LOO predictions available at this point.')

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

 
 

class NeuralGreedyV2_cp(BanditAlgorithm):
    def __init__(self, hparams, res_dir,B,  update_freq=1, name='NeuralGreedyV2_cp'):
        self.name = name 
        self.hparams = hparams 
        self.update_freq = update_freq 
        opt = optax.adam(hparams.lr)
        self.nn = NeuralBanditModelV2(opt, hparams, '{}_nn2'.format(name))
        self.data = BanditDataset(hparams.context_dim, hparams.num_actions, hparams.buffer_s, '{}-data'.format(name))
        self.prediction_interval_model = None
        self.res_dir  = res_dir
        self.Ensemble_pred_interval_centers = []   
        self.B =B

    def reset(self, seed): 
        self.nn.reset(seed) 
        self.data.reset()

    def sample_action(self, contexts, opt_vals, opt_actions):
        preds = []
        for a in range(self.hparams.num_actions):
            actions = jnp.ones(shape=(contexts.shape[0],)) * a 
            # ====== added by Annie 2024 Oct.04
            if len(self.Ensemble_pred_interval_centers) == 0:
                f = self.nn.out(self.nn.params, contexts, actions)  # Default to the neural network output
            else:
                f = self.Ensemble_pred_interval_centers
                f = jnp.array(f)
            # ====== added by Annie 2024 Oct.04

            # f = self.nn.out(self.nn.params, contexts, actions) # (num_samples, 1)
            preds.append(f) 
        '''
        === debug log ====
        jnp.hstack flattens the array along the horizontal axis, resulting in a one-dimensional array. This is why you end up with preds.shape == (1040,) instead of the expected two dimensions.
        Since you have 520 contexts and are making predictions for 2 actions, jnp.hstack concatenates them into a single flat array instead of keeping them as separate rows and columns.
        Solution:
        To fix this, you should use jnp.column_stack (or jnp.stack) instead of jnp.hstack. jnp.column_stack will ensure that preds has two dimensions: one for the number of contexts and another for the number of actions.
        '''
        # preds = jnp.hstack(preds) 
        # Use column_stack to ensure that each action's predictions are in separate columns
        preds = jnp.column_stack(preds)
        # print(f"preds.shape: {preds.shape}")
        sampled_test_actions = jnp.argmax(preds, axis=1)


        X_train = self.data.contexts
        # selected actions for training
        actions = self.data.actions.ravel().astype(int)
        # rewards for training
        Y_train = self.data.rewards[np.arange(len(actions)), actions].reshape(-1, 1)
        X_predict = contexts
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

        return sampled_test_actions

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
        # for param in jax.tree_leaves(self.nn.params):
        #     print(f'param.shape in jax.tree_leaves(self.nn.params):')
        #     print(param.shape)
        # sys.exit()


        # norm = jnp.hstack(( jnp.ravel(param) for param in jax.tree_leaves(self.nn.params)))
        norm = jnp.hstack([jnp.ravel(param) if param.shape != (1,) else param.reshape(1,) for param in jax.tree_leaves(self.nn.params)])

        # convoluted_contexts = self.nn.action_convolution(contexts, actions)

        # preds = self.nn.out(self.nn.params, contexts, actions) # (num_samples,)

        preds = []
        for a in range(self.hparams.num_actions):
            actions_tmp = jnp.ones(shape=(contexts.shape[0],)) * a 

            if len(self.Ensemble_pred_interval_centers) == 0:
                f = self.nn.out(self.nn.params, contexts, actions_tmp) # (num_samples, 1)
            else:
                f = self.Ensemble_pred_interval_centers
                f = jnp.array(f)


            # f = self.nn.out(self.nn.params, contexts, actions_tmp) # (num_samples, 1)
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

class NeuraLCB_cp(BanditAlgorithm):
    """NeuraLCB using diag approximation for confidence matrix. """
    def __init__(self, hparams, res_dir, update_freq, name='NeuraLCB_cp'):
        self.name = name 
        self.hparams = hparams 
        opt = optax.adam(hparams.lr) 
        # here is why NeuraLCB is different from ApproxNeuraLCBV2
         
        self.nn = NeuralBanditModel(opt, hparams, '{}_nn'.format(self.name))
        self.data = BanditDataset(hparams.context_dim, hparams.num_actions, hparams.buffer_s, 'bandit_data')
        self.diag_Lambda = jnp.array([hparams.lambd0 * jnp.ones(self.nn.num_params) for _ in range(self.hparams.num_actions)]) # (num_actions, p)
        self.update_freq = update_freq
        self.prediction_interval_model = None
        self.res_dir  = res_dir
        self.Ensemble_pred_interval_centers = []  
    def sample_action(self, contexts,opt_vals, opt_actions):
        """
        Args:
            contexts: (None, self.hparams.context_dim)
        """



        n = contexts.shape[0] 
        cs = self.hparams.chunk_size
        num_chunks = math.ceil(contexts.shape[0] / cs)
        acts = []


        for i in range(num_chunks):
            ctxs = contexts[i * cs: (i+1) * cs,:] 
            lcb = []
          
            # actions = jnp.ones(shape=(ctxs.shape[0],)) * a 
            # this is the predicted rewards 
            start_idx = i * cs
            end_idx = min((i + 1) * cs, n)
            ctxs = contexts[start_idx:end_idx, :]  # Shape (batch_size, context_dim)
            

            if len(self.Ensemble_pred_interval_centers) == 0:
                f = self.nn.out(self.nn.params, ctxs)  # Default to the neural network output
            else:             
                f = self.Ensemble_pred_interval_centers[start_idx:end_idx]
                f = jnp.array(f)
                print(f'self.Ensemble_pred_interval_centers available!!!!')
                print(f'self.Ensemble_pred_interval_centers.shape == {self.Ensemble_pred_interval_centers.shape}')
            # g = self.nn.grad_out(self.nn.params, convoluted_contexts) / jnp.sqrt(self.nn.m) # (num_samples, p)
            # g = self.nn.grad_out(self.nn.params, ctxs, actions) / jnp.sqrt(self.nn.m)
            
            g = self.nn.grad_out(self.nn.params, ctxs)  # Shape (num_actions, batch_size, context_dim)
            
            cnf = jnp.sqrt(jnp.sum(jnp.square(g) / self.diag_Lambda.reshape(self.hparams.num_actions, 1, -1), axis=-1)) / jnp.sqrt(self.nn.m)
            cnf = cnf.T

            # lcb= f.ravel() - self.hparams.beta * cnf.ravel()  # (num_samples,)
            lcb = f - self.hparams.beta * cnf 
            print(f'lcb.shape == {lcb.shape}')
            print(f'cnf.shape == {cnf.shape}')
            
   
            acts.append(jnp.argmax(lcb, axis=1)) 

        sampled_test_actions = jnp.concatenate(acts)
        print(f'len(sampled_test_actions) == {len(sampled_test_actions)}')

        print(f'sample_test_actions.shape==={sampled_test_actions.shape}')

        assert sampled_test_actions.shape[0] == n, f"Mismatch between number of contexts ({n}) and sampled actions ({sampled_test_actions.shape[0]})"


        X_train = self.data.contexts
        # selected actions for training
        actions = self.data.actions.ravel().astype(int)
        # rewards for training
        Y_train = self.data.rewards[np.arange(len(actions)), actions].reshape(-1, 1)
        X_predict = contexts
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
                        self.name)
        self.Ensemble_pred_interval_centers = self.prediction_interval_model.fit_bootstrap_models_online(B=10, miss_test_idx=[])
        # print(f'self.Ensemble_prediction_interval_centers:{self.Ensemble_pred_interval_centers}')
        PI_dfs, results = self.prediction_interval_model.run_experiments(alpha=0.05, stride=8,methods=['Ensemble'])    
            
        return sampled_test_actions
        
            
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
        # no actions are involved in the updating process. differentfrom Approxlcbv2
        g = self.nn.grad_out(self.nn.params, contexts)  # (num_actions, num_samples, p)
        g = jnp.square(g) / self.nn.m 
        for i in range(g.shape[1]): 
            self.diag_Lambda += g[:,i,:]

    def monitor(self, contexts=None, actions=None, rewards=None):
        # params = jnp.hstack(( jnp.ravel(param) for param in jax.tree_leaves(self.nn.params)))
        params = jnp.hstack([jnp.ravel(param) if param.shape != (1,) else param.reshape(1,) for param in jax.tree_leaves(self.nn.params)])
 


        
        if len(self.Ensemble_pred_interval_centers) == 0:
            f = self.nn.out(self.nn.params, contexts) # (num_samples, num_actions)
        else:
            f = self.Ensemble_pred_interval_centers
            f = jnp.array(f)
        g = self.nn.grad_out(self.nn.params, contexts)
        cnf = jnp.sqrt( jnp.sum(jnp.square(g) / self.diag_Lambda.reshape(self.hparams.num_actions,1,-1), axis=-1) ) / jnp.sqrt(self.nn.m)

        


        
        # action and reward fed here are in vector forms, we convert them into an array of one-hot vectors
        action_hot = jax.nn.one_hot(actions.ravel(), self.hparams.num_actions) 
        reward_hot = action_hot * rewards.reshape(-1,1)

        cost = self.nn.loss(self.nn.params, contexts, action_hot, reward_hot)
        print(f'~~~~~~~~~~~~~~~~~~~========================~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
        print('     r: {} | a: {} | f: {} | cnf: {} | loss: {}  | param_mean: {}'.format(rewards.ravel()[0], actions.ravel()[0], f.ravel(), \
            cnf.ravel(), cost, jnp.mean(jnp.square(params))))
        print(f'~~~~~~~~~~~~~~~~~~~========================~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')

 

class ExactNeuraLCB(BanditAlgorithm):
    """NeuraLCB using exact confidence matrix. """
    def __init__(self, hparams, res_dir,name='ExactNeuraLCB'):
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
        self.prediction_interval_model = None
        self.res_dir  = res_dir
        self.Ensemble_fitted_func = []
        self.Ensemble_online_resid = np.array([])
        self.Ensemble_pred_interval_centers = []   
        self.Ensemble_train_interval_centers = []  # Predicted training data centers by EnbPI
        self.beta_hat_bins = []


    def sample_action_original(self, context):
        """
        Args:
            context: (None, self.hparams.context_dim)
        """
        n = context.shape[0] 
        
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
            
    def sample_action(self, context, opt_vals, opt_actions):
        """
        Args:
            context: (None, self.hparams.context_dim)
        """
        n = context.shape[0] 
        
        if n <= self.hparams.max_test_batch:
            if len(self.Ensemble_pred_interval_centers) == 0:
                f = self.nn.out(self.nn.params, context) # (num_samples, num_actions)
            else:
                f = self.Ensemble_pred_interval_centers 

            g = self.nn.grad_out(self.nn.params, context) / jnp.sqrt(self.nn.m) # (num_actions, num_samples, p)
            gA = jnp.sum(jnp.multiply(g[:,:,None,:], self.Lambda_inv[:, None, :,:]), axis=-1) # (num_actions, num_samples, p)
            gAg = jnp.sum(jnp.multiply(gA, g), axis=-1) # (num_actions, num_samples)
            cnf = jnp.sqrt(gAg).T # (num_samples, num_actions)

            lcb = f - self.hparams.beta * cnf   # (num_samples, num_actions)
            sampled_test_actions = jnp.argmax(lcb, axis=1)
            # return jnp.argmax(lcb, axis=1)
        else: # Break contexts in batches if it is large.
            inv = int(n / self.hparams.max_test_batch)
            acts = []
            for i in range(inv):
                c = context[i*self.hparams.max_test_batch:self.hparams.max_test_batch*(i+1),:]
                if len(self.Ensemble_pred_interval_centers) == 0:
                    f = self.nn.out(self.nn.params, c) # (num_samples, num_actions)
                else:
                
                    f = self.Ensemble_pred_interval_centers[i*self.hparams.max_test_batch:self.hparams.max_test_batch*(i+1)]
                    f = jnp.array(f)
                    print(f'!!!!!! Prediction Intervals available!!!!')
                    print(f'&&&&& len(Ensemble_pred_interval_centers)  === {len(self.Ensemble_pred_interval_centers)}&&&&&')
                    print(f'PI centers datatype == {type(self.Ensemble_pred_interval_centers)}')
                # f = self.nn.out(self.nn.params, c) # (num_samples, num_actions)

                g = self.nn.grad_out(self.nn.params, c) # (num_actions, num_samples, p)

                gA = jnp.sum(jnp.multiply(g[:,:,None,:], self.Lambda_inv[:, None, :,:]), axis=-1) # (num_actions, num_samples, p)
                gAg = jnp.sum(jnp.multiply(gA, g), axis=-1) # (num_actions, num_samples)
                cnf = jnp.sqrt(gAg).T # (num_samples, num_actions)

                lcb = f - self.hparams.beta * cnf   # (num_samples, num_actions)
                acts.append(jnp.argmax(lcb, axis=1).ravel())
            sampled_test_actions = jnp.hstack(acts)
            # return jnp.array(acts)
        

        X_train = self.data.contexts
        # selected actions for training
        actions = self.data.actions.ravel().astype(int)
        # rewards for training
        Y_train = self.data.rewards[np.arange(len(actions)), actions].reshape(-1, 1)
        X_predict = context
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
                        filename)
        self.Ensemble_pred_interval_centers = self.prediction_interval_model.fit_bootstrap_models_online(B=10, miss_test_idx=[])
        print(f'self.Ensemble_prediction_interval_centers:{self.Ensemble_pred_interval_centers}')
        PI_dfs, results = self.prediction_interval_model.run_experiments(alpha=0.05, stride=8,methods=['Ensemble'])       
        
        # return jnp.hstack(acts)
        return sampled_test_actions
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
        self.nn = NeuralBanditModel(opt, hparams, '{}_nn'.format(self.name))
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


