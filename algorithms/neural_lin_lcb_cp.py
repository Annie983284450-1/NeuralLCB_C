"""Define Neural Linear LCB offline bandit. """

import jax 
import jax.numpy as jnp
import optax
import numpy as np 
import math 
from jax.scipy.linalg import cho_factor, cho_solve
from core.bandit_algorithm import BanditAlgorithm 
from core.utils import inv_sherman_morrison, inv_sherman_morrison_single_sample, vectorize_tree
from algorithms.neural_bandit_model import NeuralBanditModel, NeuralBanditModelV2
import importlib
import cp_funs.PI
importlib.reload(cp_funs.PI)
from cp_funs.PI import prediction_interval
from core.bandit_dataset import BanditDataset



# ======================== ========================ExactNeuralLinLCBV2_cp================================================
class ExactNeuralLinLCBV2_cp(BanditAlgorithm):
    def __init__(self, hparams, res_dir, B,  update_freq=1, name='ExactNeuralLinLCBV2_cp'):
        self.name = name 
        self.hparams = hparams 
        self.update_freq = update_freq 
        # opt = optax.adam(0.0001) # dummy
        opt = optax.adam(hparams.lr)
        self.nn = NeuralBanditModelV2(opt, hparams, '{}_nn2'.format(name))
        
        self.reset(self.hparams.seed)
        self.data = BanditDataset(hparams.context_dim, hparams.num_actions, hparams.buffer_s, '{}-data'.format(name))
        self.prediction_interval_model = None
        self.res_dir  = res_dir
        self.Ensemble_pred_interval_centers = []   
        self.B = B

    def reset(self, seed): 
        self.y_hat = jnp.zeros(shape=(self.hparams.num_actions, self.nn.num_params))
        # This snippet represents Lambda_inv as a full matrix for each action
        self.Lambda_inv = jnp.array(
            [
                jnp.eye(self.nn.num_params)/ self.hparams.lambd0 for _ in range(self.hparams.num_actions)
            ]
        ) # (num_actions, p, p)

        self.nn.reset(seed) 

    def sample_action(self, contexts,opt_vals, opt_actions,res_dir, algo_prefix):
        cs = self.hparams.chunk_size
        num_chunks = math.ceil(contexts.shape[0] / cs)
        acts = []
        for i in range(num_chunks):
            ctxs = contexts[i * cs: (i+1) * cs,:] 
            lcb = []
            for a in range(self.hparams.num_actions):
                actions = jnp.ones(shape=(ctxs.shape[0],)) * a 

                g = self.nn.grad_out(self.nn.params, ctxs, actions) / jnp.sqrt(self.nn.m) # (None, p)
                gA = g @ self.Lambda_inv[a,:,:] # (num_samples, p)
                
                gAg = jnp.sum(jnp.multiply(gA, g), axis=-1) # (num_samples, )
                cnf = jnp.sqrt(gAg) # (num_samples,)

                # f = jnp.dot(gA, self.y_hat[a,:]) 

                if len(self.Ensemble_pred_interval_centers) == 0:
                    # Original calculation for `f` (using `y_hat` and `diag_Lambda` for exploration-exploitation)
                    f = jnp.dot(gA, self.y_hat[a,:]) 
                else:
                    # Modify `f` using the conformal prediction interval as an additional confidence adjustment
                    conformal_center = self.Ensemble_pred_interval_centers[i * cs: (i+1) * cs]
                    f_conformal = jnp.array(conformal_center)
                    # Combining conformal center with original `f` to maintain uncertainty control
                    f = f_conformal + (jnp.dot(gA, self.y_hat[a,:]) - f_conformal) * 0.5


                lcb_a = f.ravel() - self.hparams.beta * cnf.ravel()  # (num_samples,)
                lcb.append(lcb_a.reshape(-1,1)) 
            lcb = jnp.hstack(lcb) 
            acts.append( jnp.argmax(lcb, axis=1)) 
        sampled_test_actions = jnp.hstack(acts)
        

        # computing preidcition intervals
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
        alphacp_ls = np.linspace(0.05,0.25,5)
        for alphacp in alphacp_ls:
            PI_dfs, results = self.prediction_interval_model.run_experiments(alpha=alphacp, stride=8,methods=['Ensemble'],res_dir=res_dir, algo_prefix=algo_prefix)       
        
        
        # return jnp.hstack(acts)
        return sampled_test_actions
    
    

    def update_buffer(self, contexts, actions, rewards): 
        self.data.add(contexts, actions, rewards)



    def update(self, contexts, actions, rewards): 
        u = self.nn.grad_out(self.nn.params, contexts, actions) / jnp.sqrt(self.nn.m)

        for i in range(contexts.shape[0]):
            # jax.ops.index_update has been deprecated in recent versions of jax.
            # jax.ops.index_update(self.y_hat, actions[i], \
            #     self.y_hat[actions[i]] +  rewards[i] * u[i,:])
            # jax.ops.index_update(self.Lambda_inv, actions[i], \
            #     inv_sherman_morrison_single_sample(u[i,:], self.Lambda_inv[actions[i],:,:]))

            self.y_hat = self.y_hat.at[actions[i]].set(
                self.y_hat[actions[i]] + rewards[i] * u[i, :]
            )
            self.Lambda_inv = self.Lambda_inv.at[actions[i]].set(
                inv_sherman_morrison_single_sample(u[i, :], self.Lambda_inv[actions[i], :, :])
            )

    def monitor(self, contexts=None, actions=None, rewards=None):

        # norm = jnp.hstack(( jnp.ravel(param) for param in jax.tree_leaves(self.nn.params)))
        norm = jnp.hstack([jnp.ravel(param) if param.shape != (1,) else param.reshape(1,) for param in jax.tree_leaves(self.nn.params)])

        preds = []
        cnfs = []
        for a in range(self.hparams.num_actions):
            actions_tmp = jnp.ones(shape=(contexts.shape[0],)) * a 
            g = self.nn.grad_out(self.nn.params, contexts, actions_tmp) / jnp.sqrt(self.nn.m) # (None, p)
            gA = g @ self.Lambda_inv[a,:,:] # (num_samples, p)
            gAg = jnp.sum(jnp.multiply(gA, g), axis=-1) # (num_samples, )
            cnf = jnp.sqrt(gAg) # (num_samples,)
            if len(self.Ensemble_pred_interval_centers) == 0:
                # Original calculation for `f` (using `y_hat` and `diag_Lambda` for exploration-exploitation)
                f = jnp.dot(gA, self.y_hat[a,:]) 
            else:
                # Modify `f` using the conformal prediction interval as an additional confidence adjustment
                conformal_center = self.Ensemble_pred_interval_centers 
                f_conformal = jnp.array(conformal_center)
                # Combining conformal center with original `f` to maintain uncertainty control
                f = f_conformal + (jnp.dot(gA, self.y_hat[a,:]) - f_conformal) * 0.5

            cnfs.append(cnf) 
            preds.append(f)
        cnf = jnp.hstack(cnfs) 
        preds = jnp.hstack(preds)

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
            


# ======================== ========================ApproxNeuralLinLCBV2_cp================================================


class ApproxNeuralLinLCBV2_cp(BanditAlgorithm):
    def __init__(self, hparams, res_dir, B, update_freq=1, name='ApproxNeuralLinLCBV2_cp'):
        self.name = name 
        self.hparams = hparams 
        self.update_freq = update_freq 
        # opt = optax.adam(0.0001) # dummy
        opt = optax.adam(hparams.lr)
        self.nn = NeuralBanditModelV2(opt, hparams, '{}_nn2'.format(name))
        self.data = BanditDataset(hparams.context_dim, hparams.num_actions, hparams.buffer_s, '{}-data'.format(name))
        self.prediction_interval_model = None
        self.res_dir  = res_dir
        self.Ensemble_pred_interval_centers = []   
        self.reset(self.hparams.seed)
        self.B = B

    def reset(self, seed): 
        # self.y_hat = jnp.zeros(shape=(self.hparams.num_actions, self.nn.num_params))

        self.y_hat = [
            jnp.zeros(shape=(self.nn.num_params,)) for _ in range(self.hparams.num_actions)
        ]
        # This snippet represents diag_Lambda as a diagonal-only representation for each action.
        self.diag_Lambda = [
                jnp.ones(self.nn.num_params) * self.hparams.lambd0 for _ in range(self.hparams.num_actions)
            ]
         # (num_actions, p)

        self.nn.reset(seed) 

    def sample_action(self, contexts,opt_vals, opt_actions, res_dir, algo_prefix):
        cs = self.hparams.chunk_size
        num_chunks = math.ceil(contexts.shape[0] / cs)
        acts = []
        for i in range(num_chunks):
            ctxs = contexts[i * cs: (i+1) * cs,:] 
            lcb = []
            for a in range(self.hparams.num_actions):
                actions = jnp.ones(shape=(ctxs.shape[0],)) * a 

                g = self.nn.grad_out(self.nn.params, ctxs, actions) / jnp.sqrt(self.nn.m) # (None, p)

                gAg = jnp.sum(jnp.square(g) / self.diag_Lambda[a][:], axis=-1)                
                cnf = jnp.sqrt(gAg) # (num_samples,)

                if len(self.Ensemble_pred_interval_centers) == 0:
                    # Original calculation for `f` (using `y_hat` and `diag_Lambda` for exploration-exploitation)
                    f = jnp.sum(jnp.multiply(g, self.y_hat[a][:]) / self.diag_Lambda[a][:], axis=-1)
                else:
                    # Modify `f` using the conformal prediction interval as an additional confidence adjustment
                    conformal_center = self.Ensemble_pred_interval_centers[i * cs: (i+1) * cs]
                    f_conformal = jnp.array(conformal_center)
                    # Combining conformal center with original `f` to maintain uncertainty control
                    f = f_conformal + (jnp.sum(jnp.multiply(g, self.y_hat[a][:]) / self.diag_Lambda[a][:], axis=-1) - f_conformal) * 0.5

                # f = jnp.sum(jnp.multiply(g, self.y_hat[a][:]) / self.diag_Lambda[a][:], axis=-1)

                lcb_a = f.ravel() - self.hparams.beta * cnf.ravel()  # (num_samples,)
                lcb.append(lcb_a.reshape(-1,1)) 
            lcb = jnp.hstack(lcb) 
            acts.append( jnp.argmax(lcb, axis=1)) 
        sampled_test_actions = jnp.hstack(acts)
        

        # computing preidcition intervals
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
        alphacp_ls = np.linspace(0.05,0.25,5)
        for alphacp in alphacp_ls:
            PI_dfs, results = self.prediction_interval_model.run_experiments(alpha=alphacp, stride=8,methods=['Ensemble'],res_dir=res_dir, algo_prefix=algo_prefix)       
       
        
        # return jnp.hstack(acts)
        return sampled_test_actions
    


    def update_buffer(self, contexts, actions, rewards): 
        self.data.add(contexts, actions, rewards)

    def update(self, contexts, actions, rewards): 
        # print(rewards)
        u = self.nn.grad_out(self.nn.params, contexts, actions) / jnp.sqrt(self.nn.m)
        for i in range(contexts.shape[0]):
            v = self.diag_Lambda[actions[i]] 
            # jax.ops.index_update(self.diag_Lambda, actions[i], \
            #     jnp.square(u[i,:]) + self.diag_Lambda[actions[i],:])  
            self.diag_Lambda[actions[i]] = jnp.square(u[i,:]) + self.diag_Lambda[actions[i]]

            self.y_hat[actions[i]] = self.y_hat[actions[i]] +  rewards[i] * u[i,:] 
            # jax.ops.index_add(self.y_hat, actions[i], rewards[i] * u[i,:])

    def monitor(self, contexts=None, actions=None, rewards=None):

        # norm = jnp.hstack(( jnp.ravel(param) for param in jax.tree_leaves(self.nn.params)))
        norm = jnp.hstack([jnp.ravel(param) if param.shape != (1,) else param.reshape(1,) for param in jax.tree_leaves(self.nn.params)])

        preds = []
        cnfs = []
        for a in range(self.hparams.num_actions):
            actions_tmp = jnp.ones(shape=(contexts.shape[0],)) * a 

            g = self.nn.grad_out(self.nn.params, contexts, actions_tmp) / jnp.sqrt(self.nn.m) # (num_samples, p)
            gAg = jnp.sum(jnp.square(g) / self.diag_Lambda[a][:], axis=-1)                
            cnf = jnp.sqrt(gAg) # (num_samples,)
            if len(self.Ensemble_pred_interval_centers) == 0:
                # Original calculation for `f` (using `y_hat` and `diag_Lambda` for exploration-exploitation)
                f = jnp.sum(jnp.multiply(g, self.y_hat[a][:]) / self.diag_Lambda[a][:], axis=-1)
            else:
                # Modify `f` using the conformal prediction interval as an additional confidence adjustment
                conformal_center = self.Ensemble_pred_interval_centers
                f_conformal = jnp.array(conformal_center)
                # Combining conformal center with original `f` to maintain uncertainty control
                f = f_conformal + (jnp.sum(jnp.multiply(g, self.y_hat[a][:]) / self.diag_Lambda[a][:], axis=-1) - f_conformal) * 0.5
            # f = jnp.sum(jnp.multiply(g, self.y_hat[a][:]) / self.diag_Lambda[a][:], axis=-1)

            cnfs.append(cnf) 
            preds.append(f)
        cnf = jnp.hstack(cnfs) 
        preds = jnp.hstack(preds)

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
            




class ApproxNeuralLinLCBJointModel_cp(BanditAlgorithm):
    """Use joint model as LinLCB. 
    
    Sigma_t = lambda I + \sum_{i=1}^t phi(x_i,a_i) ph(x_i,a_i)^T  
    theta_t = Sigma_t^{-1} \sum_{i=1}^t phi(x_i,a_i) y_i ""
    """
    def __init__(self, hparams, res_dir, B, update_freq=1, name='ApproxNeuralLinLCBJointModel_cp'):
        self.name = name 
        self.hparams = hparams 
        self.update_freq = update_freq 
        # opt = optax.adam(0.0001) # dummy
        opt = optax.adam(hparams.lr)
        self.nn = NeuralBanditModelV2(opt, hparams, '{}_nn2'.format(name))
        self.B = B
        
        self.reset(self.hparams.seed)
        self.prediction_interval_model = None
        self.res_dir  = res_dir
        self.Ensemble_pred_interval_centers = [] 
        self.data = BanditDataset(hparams.context_dim, hparams.num_actions, hparams.buffer_s, '{}-data'.format(name))



    def reset(self, seed): 
        # self.y_hat = jnp.zeros(shape=(self.hparams.num_actions, self.nn.num_params))
        self.y_hat = jnp.zeros(shape=(self.nn.num_params,)) 
        self.diag_Lambda = jnp.ones(self.nn.num_params) * self.hparams.lambd0  
        self.nn.reset(seed) 

    def sample_action(self, contexts, opt_vals, opt_actions, res_dir, algo_prefix):
        cs = self.hparams.chunk_size
        num_chunks = math.ceil(contexts.shape[0] / cs)
        acts = []
        for i in range(num_chunks):
            ctxs = contexts[i * cs: (i+1) * cs,:] 
            lcb = []
            for a in range(self.hparams.num_actions):
                actions = jnp.ones(shape=(ctxs.shape[0],)) * a 

                g = self.nn.grad_out(self.nn.params, ctxs, actions) / jnp.sqrt(self.nn.m) # (None, p)

                gAg = jnp.sum(jnp.square(g) / self.diag_Lambda, axis=-1)                
                cnf = jnp.sqrt(gAg) # (num_samples,)

                # f = jnp.sum(jnp.multiply(g, self.y_hat) / self.diag_Lambda, axis=-1)

                if len(self.Ensemble_pred_interval_centers) == 0:
                    # Original calculation for `f` (using `y_hat` and `diag_Lambda` for exploration-exploitation)
                    f = jnp.sum(jnp.multiply(g, self.y_hat) / self.diag_Lambda, axis=-1)

                else:
                    # Modify `f` using the conformal prediction interval as an additional confidence adjustment
                    conformal_center = self.Ensemble_pred_interval_centers[i * cs: (i+1) * cs]
                    f_conformal = jnp.array(conformal_center)
                    # Combining conformal center with original `f` to maintain uncertainty control
                    f = f_conformal + (jnp.sum(jnp.multiply(g, self.y_hat) / self.diag_Lambda, axis=-1) - f_conformal) * 0.5



                lcb_a = f.ravel() - self.hparams.beta * cnf.ravel()  # (num_samples,)
                lcb.append(lcb_a.reshape(-1,1)) 
            lcb = jnp.hstack(lcb) 
            acts.append( jnp.argmax(lcb, axis=1)) 
        # return jnp.hstack(acts)
        sampled_test_actions = jnp.hstack(acts)
        

        # computing preidcition intervals
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
        alphacp_ls = np.linspace(0.05,0.25,5)
        for alphacp in alphacp_ls:
            PI_dfs, results = self.prediction_interval_model.run_experiments(alpha=alphacp, stride=8,methods=['Ensemble'],res_dir=res_dir, algo_prefix=algo_prefix)       
    
        # return jnp.hstack(acts)
        return sampled_test_actions
    

    def update_buffer(self, contexts, actions, rewards): 
        self.data.add(contexts, actions, rewards)

    def update(self, contexts, actions, rewards): 
        # print(rewards)
        u = self.nn.grad_out(self.nn.params, contexts, actions) / jnp.sqrt(self.nn.m)
        for i in range(contexts.shape[0]):
            # jax.ops.index_update(self.diag_Lambda, actions[i], \
            #     jnp.square(u[i,:]) + self.diag_Lambda[actions[i],:])  
            self.diag_Lambda = jnp.square(u[i,:]) + self.diag_Lambda
            self.y_hat = self.y_hat +  rewards[i] * u[i,:] 
        
 
    def monitor(self, contexts=None, actions=None, rewards=None):

        # norm = jnp.hstack(( jnp.ravel(param) for param in jax.tree_leaves(self.nn.params)))
        norm = jnp.hstack([jnp.ravel(param) if param.shape != (1,) else param.reshape(1,) for param in jax.tree_leaves(self.nn.params)])

        preds = []
        cnfs = []
        for a in range(self.hparams.num_actions):
            actions_tmp = jnp.ones(shape=(contexts.shape[0],)) * a 

            g = self.nn.grad_out(self.nn.params, contexts, actions_tmp) / jnp.sqrt(self.nn.m) # (None, p)

            gAg = jnp.sum(jnp.square(g) / self.diag_Lambda, axis=-1)                
            cnf = jnp.sqrt(gAg) # (num_samples,)

            # f = jnp.sum(jnp.multiply(g, self.y_hat) / self.diag_Lambda, axis=-1) 
 

            if len(self.Ensemble_pred_interval_centers) == 0:
              
                f = jnp.sum(jnp.multiply(g, self.y_hat) / self.diag_Lambda, axis=-1)
            else:
                # Modify `f` using the conformal prediction interval as an additional confidence adjustment
                conformal_center = self.Ensemble_pred_interval_centers
                f_conformal = jnp.array(conformal_center)
                 
                f = f_conformal + (jnp.sum(jnp.multiply(g, self.y_hat) / self.diag_Lambda, axis=-1) - f_conformal) * 0.5
            cnfs.append(cnf) 
            preds.append(f)
        cnf = jnp.hstack(cnfs) 
        preds = jnp.hstack(preds)

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
            
