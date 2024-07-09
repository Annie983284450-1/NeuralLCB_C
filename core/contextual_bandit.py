"""Define an contextual bandit class. """
# this module is only called in the main function, to get the final regrets
import numpy as np
from tqdm import tqdm
from timeit import timeit 
import time 
import sys
import PI_Sepsysolcp as EnbPI

def action_stats(actions, num_actions):
    """Compute the freq of each action.

    Args:
        actions: (None, )
        num_actions: int 
    """

    stats = [] 
    for a in range(num_actions):
        stats.append(np.mean(np.asarray(actions == a).astype('float32')))
    return stats
    
def action_accuracy(pred_actions, opt_actions):
    """Compute accuracy between predicted actions and optimal actions. 

    Args:
        pred_actions: (None,)
        opt_actions: (None,)
    """
    return np.mean(np.asarray(pred_actions == opt_actions).astype('float32'))

# def reset_data(sim):
#     # to be added
#     pass

def generate_dynamic_loo_intervals(self, X_train, Y_train, X_predict, Y_predict, expert, final_result_path, alpha, max_hours, B,Isrefit,stride, data_name, itrial):
    cp_EnbPI = EnbPI.prediction_interval(locals()[f'{expert}_f'], X_train, X_predict, Y_train, Y_predict, final_result_path)
    cp_EnbPI.fit_bootstrap_models_online_multi(B, self.boot_samples_idx, Isrefit, model_name=expert, max_hours=max_hours)
    
    # Perform leave-one-out bootstrap dynamically
    PIs_df, results = cp_EnbPI.run_experiments(alpha, stride, data_name, itrial, true_Y_predict=[], get_plots=False, none_CP=False, methods=methods, max_hours=max_hours)
    return PIs_df, results.mean_coverage.values[0]




# this is the final function 
def contextual_bandit_runner(algos, data, \
            num_sim, update_freq, test_freq, verbose, debug, normalize, save_path=None):
    """Run an offline contextual bandit problem on a set of algorithms in the same dataset. 

    Args:
        dataset: A tuple of (contexts, actions, mean_rewards, test_contexts, test_mean_rewards).
        algos: A list of algorithms to run on the bandit instance.  
    """

    # Create a bandit instance 
    regrets = [] # (num_sim, num_algos, T) 
    errs = [] # (num_sim, num_algos, T) 
    for sim in range(num_sim):
        # Run the offline contextual bandit in an online manner 
        print('Simulation: {}/{}'.format(sim + 1, num_sim))
        # reset_data() is defined in realworld_data.py in the dataclass
        # Note: data is a class not sth as a numpy array !!
        # reset_data() will return a form of (contexts, actions, rewards, test_contexts, mean_test_rewards) 
        
        cmab = OfflineContextualBandit(*data.reset_data(sim))
        # sys.exit()
     # algo: BanditAlgortihm class
        for algo in algos:
            
            #  ApproxNeuraLCBV2.reset(self, seed)
            # sim*1111 is the random seed
            algo.reset(sim * 1111)
        # initialize an empty list for each algorithm
        subopts = [
            [] for _ in range(len(algos))
        ]

        act_errs = [
            [] for _ in range(len(algos))
        ]

        # Compute test values and opt actions 
        #rewards : (num_contexts, num_actions) 
        # find the highest reward in each row(i.e., each contexr)

        # i.e., the ground truth
        opt_vals = np.max(cmab.test_mean_rewards, axis=1) 
        # find the index that leads to this optimal reward for each context (i.e., each row)
        opt_actions = np.argmax(cmab.test_mean_rewards, axis=1) 
        
        # each i is mapped to one entry in the table data
        # tqdm() generates progressive bars for loops!!!
        # it wraps around any iterable including range()
        # ncols = 75 defines the width of the bar rto 75 characters in the terminal
        # the width of the progeress bar in tdqm() is measured by characters
        # update the neural network each time there is a new context
        for i in tqdm(range(cmab.num_contexts),ncols=75):

            # start_time = timeit()
            start_time = time.time()
            # return the current state(i.e., context),action,reward & next state(i.e., ontext),action, reward
            # return self.contexts[ind:ind+1], self.actions[ind:ind+1], self.rewards[ind:ind+1, a:a+1] 
            # this is the ground truth, not the predicted value
            # Each iteration extracts a single data sample (context, action, reward) from the dataset.
            # as in pseudo code line 3
            c,a,r = cmab.get_data(i) 

            for j,algo in enumerate(algos): 
    # def update_buffer(self, contexts, actions, rewards): 
        # self.data.add(contexts, actions, rewards)
        # and add() is a self-defined calculation for the Bandit Dataset
        # Bandit Dataset is also self-defined
                algo.update_buffer(c,a,r)
                # Add data and update the internal state of each offline algorithm 
                # update_freq default value 1
                if i % algo.update_freq == 0 and algo.name != 'KernLCB':
                    # update the network parameters and confidence parameters
                    algo.update(c, a, r) 

                # Test alg 
                # test_freq default value 10
                if i % test_freq == 0:
                    if algo.name == 'KernLCB': 
                        algo.update()

                    if algo.name == 'KernLCB' and algo.X.shape[0] >= algo.hparams.max_num_sample:
                        print('KernLCB reuses the last 1000 points for prediction!')
                        test_subopt = subopts[j][-1] # reuse the last result
                        action_acc = 1 - act_errs[j][-1]
                        if verbose: # default true
                            print('[sim: {}/{} | iter: {}/{}] {} | regret: {} | acc: {} | '.format(
                                sim+1, num_sim,  i, cmab.num_contexts,
                                algo.name, test_subopt, action_acc))
                    else: 
                        t1 = time.time()
                        test_actions = algo.sample_action(cmab.test_contexts) 
                        t2 = time.time()
                        sel_vals = cmab.test_mean_rewards[np.arange(cmab.num_test_contexts), test_actions.ravel()]
                        if normalize:
                            test_subopt = np.mean(1 - sel_vals / opt_vals) 
                        else:
                            test_subopt = np.mean(opt_vals - sel_vals)
                       # action_accuracy(): return np.mean(np.asarray(pred_actions == opt_actions).astype('float32'))
                        action_acc = action_accuracy(test_actions.ravel(), opt_actions.ravel()) 

                        if verbose:  # default true
                            print('[sim: {}/{} | iter: {}/{}] {} | regret: {} | acc: {} | test_time: {} '.format(
                                sim+1, num_sim,  i, cmab.num_contexts,
                                algo.name, test_subopt, action_acc, t2-t1))
                            if debug:  # default true
                                sel_stats = action_stats(test_actions.ravel(), cmab.num_actions) 
                                opt_stats = action_stats(opt_actions.ravel(), cmab.num_actions)
                                print('     opt_rate: {} | pred_rate: {}'.format(opt_stats, sel_stats))
                                ### @@@@ bugs came from here: IndexError: list index out of range
                                # monitor(context, action, reward)
                                algo.monitor(c, a, r)

                    subopts[j].append(test_subopt) 
                    act_errs[j].append(1 - action_acc) 
        
            # time_elapsed = timeit() - start_time
            time_elapsed = time.time() - start_time
            if i % test_freq == 0:
                if verbose:
                    print('Time elapse per iteration: {}'.format(time_elapsed))
                    print('================')
                    
        regrets.append(np.array(subopts)) 
        errs.append(np.array(act_errs)) 

        if save_path: # save for every simulation
            np.savez(save_path, np.array(regrets), np.array(errs) ) 

    return np.array(regrets), np.array(errs) 


class OfflineContextualBandit(object):
    def __init__(self, contexts, actions, rewards, test_contexts, test_mean_rewards):
        """
        Args:
            contexts: (None, context_dim) 
            actions: (None,) 
            mean_rewards: (None, num_actions) 
            test_contexts: (None, context_dim)
            test_mean_rewards: (None, num_actions)
        """
        self.contexts = contexts
        self.actions = actions
        self.rewards = rewards
        self.test_contexts = test_contexts
        self.test_mean_rewards = test_mean_rewards
        self.order = range(self.num_contexts) 

    '''
    shuffle the order in which contexts (and their associated rewards and actions) are presented
    to the bandit algorithms. simulte a more realistic scenario where the order of the encountering
    different conexts is not fixed but random. 

    We cannot do this on Sepsis dataset, for there might be data leakage??
    '''
    def reset_order(self): 
        # np.permutation ensures that each interger is unique
        # range from 0 to num_contexts - 1
        self.order = np.random.permutation(self.num_contexts)
    # def reset_order(self, sim): 
        # np.random.seed(sim)
    #     self.order = np.random.permutation(self.num_contexts)

    def get_data(self, number): 
        ind = self.order[number]
        a = self.actions[ind]
        return self.contexts[ind:ind+1], self.actions[ind:ind+1], self.rewards[ind:ind+1, a:a+1] 

 
        
        
    @property 
    def num_contexts(self): 
        return self.contexts.shape[0] 

    @property 
    def num_actions(self):
        return self.test_mean_rewards.shape[1] 
    
    @property  
    def context_dim(self):
        return self.contexts.shape[1]

    @property 
    def num_test_contexts(self): 
        return self.test_contexts.shape[0]



