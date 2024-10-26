import sys
import os 
 

if os.name == 'nt':
    # Windows
    neruallcb_path = r'E:\phd thesis ideas\NeuralLCB_C'
elif os.name == 'posix':
    if sys.platform == 'darwin':
        # macOS
        neruallcb_path = '/Users/anniezhou/Desktop/NeuralLCB_C'
    elif 'el9' in os.uname().release:
        # Red Hat Enterprise Linux 9
        neruallcb_path = '/storage/home/hcoda1/6/azhou60/path_to_neurallcb_on_redhat'
    else:
        # Generic Linux case if needed
        neruallcb_path = '/path/to/neurallcb_on_generic_linux'


if neruallcb_path not in sys.path:
    sys.path.append(neruallcb_path)
import numpy as np 
import pandas as pd 
from core.utils import sample_offline_policy
# import tensorflow as tf
from torchvision.datasets import MNIST
from torchvision import transforms

 
def one_hot(df, cols):
    """Returns one-hot encoding of DataFrame df including columns in cols."""
    for col in cols:
        dummies = pd.get_dummies(df[col], prefix=col, drop_first=False)
        df = pd.concat([df, dummies], axis=1)
        df = df.drop(col, axis=1)
    return df

class MnistData(object):
    def __init__(self,
            num_contexts, 
            num_test_contexts, 
            context_dim = 784, 
            num_actions = 10, 
            noise_std = 0, 
            pi = 'eps-greedy', 
            eps = 0.1, 
            subset_r = 0.5,
            # remove_underrepresented=False, 
            remove_underrepresented=True, 
        ):
 
        self.name = 'mnist'
        # transform=transforms.ToTensor(): applies a transformation that converts images to Pytorch tensors
        dataset = MNIST('data/',train=True,transform=transforms.ToTensor(),download=True)
        test_dataset = MNIST('data/',train=False,transform=transforms.ToTensor(),download=True)

        ## Meta-info 
        self.num_contexts = num_contexts 
        self.num_test_contexts = num_test_contexts
        self.context_dim = context_dim
        self.num_actions = num_actions
        self.pi = pi #policy indicator
        self.eps = eps #epsilon value for exploration
        self.subset_r = subset_r # subset ratio
        self.noise_std = noise_std #standard deviation of noise to be added to the rewards or contexts???

        '''
        dataset.data is an attribute of the dataset instance created by MNIST class from torchvision.datasets
        this attribute contains the image data for the mnist dataset in a tensor format
        dataset.data access the raw image data stored in the dataset instance
        For MNIST dataset, tensor shape is (60000, 28, 28) for the training set, 28X28 = 784 pixels
        That's why context_dim = 784

        .view([-1, context_dim]) reshapes the data into a 2D tensor where each row represents 
        a flattened image (or context in this case). 

        The -1 infers the size of the first dimension based on the total size of 
        the tensor and the second dimension, so that the total number of elements remains the same 
        view() is used to reshape a tensor without changing its data, specific for Pytorch
        

        .numpy() creats a numpy array that shares memory with the original pytorch tensor

        changes to the tensor will reflect in the numpy array and vice versa
        '''
        contexts = dataset.data.view([-1,context_dim]).numpy()
        print(f'type(contexts) of mnist: {type(contexts)}') # numpy array
        print(f'contexts: {contexts}') ## seems to be zero matrixs
        # 'dataset.target': access the labels or target tensor of the dataset 
        # this will contain the class labels for each image in the dataset
        labels = dataset.targets.numpy()
        if remove_underrepresented:
            contexts, labels = remove_underrepresented_classes(contexts, labels)
        self.contexts, self.mean_rewards = classification_to_bandit_problem(contexts, labels, num_actions)

        # test
        test_contexts = test_dataset.data.view([-1,context_dim]).numpy()
        test_labels = test_dataset.targets.numpy()
        if remove_underrepresented:
            test_contexts, test_labels = remove_underrepresented_classes(test_contexts, test_labels)
        self.test_contexts, self.test_mean_rewards = classification_to_bandit_problem(test_contexts, test_labels, num_actions)

        reward_type = 'deterministic'
        print('{}: num_actions: {} | total_samples: {} | context_dim: {} | reward_type: {}'.format(
            self.name, num_actions, self.num_samples, context_dim, reward_type
        ))

    # @ property makes "num_samples" a read-only attribute; you can get its value but cannot set it directly
    
    @property
    def num_samples(self):
        return self.contexts.shape[0]


    @property 
    def num_test_samples(self):
        return self.test_contexts.shape[0]
    # return a dataset tailored for contextual bandit
    def reset_data(self, sim_id=0):
        # Load meta-data to generate dataset
        indices = np.load('data/meta/indices_{}.npy'.format(sim_id)) # random permutation of np.arange(100000)
        test_indices = np.load('data/meta/test_indices_{}.npy'.format(sim_id)) # random permutation of np.arange(100000)

        # Generate inds
        # this means we might have duplicate indices when len(indices) > self.num_samples
        indices = indices % self.num_samples # make sure indices fall into  [0, self.num_samples)
        test_indices = test_indices % self.num_test_samples # make sure indices fall into  [0, self.num_test_samples)
        
 

        ## this part seems to be problematic, usually we will not exceeds the self.num_samples
        # ignore this case????
        if self.num_contexts > self.num_samples:
            ind = indices[:self.num_contexts]
        else: # self.num_contexts <= self.num_samples 
            # then select self.num_contexts first distinc elements of indices

            # np.unique(indices, return_index = True) returns a tuple where the first element contains 
            # the unique indices values from "indices", 
            # [1] contains the indices of the first occurences of these unique
            # values in the original array

            # this can guarantee that we have unique inds
            print(f'num_contexts): {self.num_contexts}')
            i = np.unique(indices,return_index=True)[1]
            i.sort()
            ind = indices[i[:self.num_contexts]]
            print(f'len(ind):{len(ind)}')
        if self.num_test_contexts > self.num_samples:
            test_ind = test_indices[:self.num_test_contexts]
        else:
            i = np.unique(test_indices,return_index=True)[1]
            # sorting might be unnecessary, cuz np.unique() would already return the indices 
            i.sort()
            test_ind = test_indices[i[:self.num_test_contexts]]
            print(f'len(test_ind): {len(ind)}')
                          
        contexts, mean_rewards = self.contexts[ind,:], self.mean_rewards[ind,:] 
        test_contexts, mean_test_rewards = self.test_contexts[test_ind,:], self.test_mean_rewards[test_ind,:]

        #create rewards
        # mean_rewards are based on the action and the label, select the right label 1, otherwise 0
        # why should we add the noise???
        # This simulates a realistic scenario where the observed rewards may deviate 
        # from the expected mean due to randomness or noise in the environment.???
        rewards = mean_rewards + self.noise_std * np.random.normal(size=mean_rewards.shape)
        actions = sample_offline_policy(mean_rewards, self.num_contexts, self.num_actions, self.pi, self.eps, self.subset_r,contexts,rewards)

        dataset = (contexts, actions, rewards, test_contexts, mean_test_rewards) 
        return dataset 

class MushroomData(object):
    def __init__(self, 
                num_contexts, 
                num_test_contexts, 
                num_actions = 2, 
                r_noeat=0,
                r_eat_safe=5,
                r_eat_poison_bad=-35,
                r_eat_poison_good=5,
                prob_poison_bad=0.5,
                pi = 'eps-greedy', 
                eps = 0.1, 
                subset_r = 0.5, 
                ): 

        self.name = 'mushroom'
        filename = 'data/mushroom.data'
        filename = 'data/mushroom/agaricus-lepiota.data' 
        self.num_contexts = num_contexts 
        self.num_test_contexts = num_test_contexts
        self.num_actions = num_actions 
        self.noise_std = 0 # dummy
        self.r_noeat = r_noeat 
        self.r_eat_safe = r_eat_safe 
        self.r_eat_poison_bad = r_eat_poison_bad 
        self.r_eat_poison_good = r_eat_poison_good 
        self.prob_poison_bad = prob_poison_bad 
        self.pi = pi 
        self.eps = eps 
        self.subset_r = subset_r

        df = pd.read_csv(filename, header=None)
        self.df = one_hot(df, df.columns)

        print('Mushroom: num_samples: {}'.format(self.num_samples)) 

    @property 
    def num_samples(self): 
        return self.df.shape[0]


    def reset_data(self, sim_id=0): 
        # Load meta-data to generate dataset
        indices = np.load('data/meta/indices_{}.npy'.format(sim_id)) # random permutation of np.arange(100000)
        test_indices = np.load('data/meta/test_indices_{}.npy'.format(sim_id)) # random permutation of np.arange(100000)

        # Generate inds
        indices = indices % self.num_samples
        test_indices = test_indices % self.num_samples

        if self.num_contexts > self.num_samples:
            ind = indices[:self.num_contexts]
        else:
            # then select self.num_contexts first distinc elements of indices
            i = np.unique(indices,return_index=True)[1]
            i.sort()
            ind = indices[i[:self.num_contexts]]

        if self.num_test_contexts > self.num_samples:
            test_ind = test_indices[:self.num_test_contexts]
        else:
            i = np.unique(test_indices,return_index=True)[1]
            i.sort()
            test_ind = test_indices[i[:self.num_test_contexts]]

        contexts = self.df.iloc[ind, 2:].values 
        test_contexts = self.df.iloc[test_ind, 2:].values 


        # Compute mean rewards 
        no_eat_reward = self.r_noeat * np.ones((self.num_contexts, 1))
        mean_eat_poison_reward = self.r_eat_poison_bad * self.prob_poison_bad + self.r_eat_poison_good * (1 - self.prob_poison_bad)
        mean_eat_reward = self.r_eat_safe * self.df.iloc[ind, 0] + np.multiply(mean_eat_poison_reward, self.df.iloc[ind, 1])
        mean_eat_reward = mean_eat_reward.values.reshape((self.num_contexts, 1))
        mean_rewards = np.hstack((no_eat_reward, mean_eat_reward))

        no_eat_reward = self.r_noeat * np.ones((self.num_test_contexts, 1))
        mean_eat_reward = self.r_eat_safe * self.df.iloc[test_ind, 0] + np.multiply(mean_eat_poison_reward, self.df.iloc[test_ind, 1])
        mean_eat_reward = mean_eat_reward.values.reshape((self.num_test_contexts, 1))
        mean_test_rewards = np.hstack((no_eat_reward, mean_eat_reward))

        # create rewards
        no_eat_reward = self.r_noeat * np.ones((self.num_contexts, 1))
        random_poison = np.random.choice([self.r_eat_poison_bad, self.r_eat_poison_good],
                                p=[self.prob_poison_bad, 1 - self.prob_poison_bad], size= self.num_contexts)
        eat_reward = self.r_eat_safe * self.df.iloc[ind, 0]
        eat_reward += np.multiply(random_poison, self.df.iloc[ind, 1])
        eat_reward = eat_reward.values.reshape((self.num_contexts, 1))
        rewards = np.hstack((no_eat_reward, eat_reward))
        actions = sample_offline_policy(mean_rewards, self.num_contexts, self.num_actions, self.pi, self.eps, self.subset_r,contexts,rewards)

        dataset = (contexts, actions, rewards, test_contexts, mean_test_rewards) 
        return dataset 


class JesterData(object):
    def __init__(self,
            num_contexts,
            num_test_contexts, 
            context_dim = 32, 
            num_actions = 8, 
            noise_std = 0.1, 
            pi = 'eps-greedy', 
            eps = 0.1, 
            subset_r = 0.5
        ): 
        self.name = 'jester'
        file_name = 'data/jester.npy'
        with open(file_name, 'rb') as f:
            self.dataset = np.load(f)

        assert context_dim + num_actions == self.dataset.shape[1]

        ## Meta-info 
        self.num_contexts = num_contexts 
        self.num_test_contexts = num_test_contexts
        self.context_dim = context_dim
        self.num_actions = num_actions
        self.pi = pi 
        self.eps = eps 
        self.subset_r = subset_r
        self.noise_std = noise_std 
        reward_type = 'stochastic'
        print('{}: num_actions: {} | total_samples: {} | context_dim: {} | reward_type: {}'.format(
            'jester', num_actions, self.dataset.shape[0], context_dim, reward_type
        ))

    @property
    def num_samples(self):
        return self.dataset.shape[0]

    def reset_data(self, sim_id=0):
        # Load meta-data to generate dataset
        indices = np.load('data/meta/indices_{}.npy'.format(sim_id)) # random permutation of np.arange(100000)
        test_indices = np.load('data/meta/test_indices_{}.npy'.format(sim_id)) # random permutation of np.arange(100000)

        # Generate inds
        indices = indices % self.num_samples
        test_indices = test_indices % self.num_samples

        if self.num_contexts > self.num_samples:
            ind = indices[:self.num_contexts]
        else:
            # then select self.num_contexts first distinc elements of indices
            i = np.unique(indices,return_index=True)[1]
            i.sort()
            ind = indices[i[:self.num_contexts]]

        if self.num_test_contexts > self.num_samples:
            test_ind = test_indices[:self.num_test_contexts]
        else:
            i = np.unique(test_indices,return_index=True)[1]
            i.sort()
            test_ind = test_indices[i[:self.num_test_contexts]]

        contexts = self.dataset[ind, :self.context_dim]
        mean_rewards = self.dataset[ind, self.context_dim:] 

        test_contexts = self.dataset[test_ind, :self.context_dim]
        mean_test_rewards = self.dataset[test_ind, self.context_dim:] 

        actions = sample_offline_policy(mean_rewards, self.num_contexts, self.num_actions, self.pi, self.eps, self.subset_r)

        #create rewards
        rewards = mean_rewards + self.noise_std * np.random.normal(size=mean_rewards.shape)
        dataset = (contexts, actions, rewards, test_contexts, mean_test_rewards) 
        return dataset 




class StatlogData(object):
    def __init__(self,
            num_contexts, 
            num_test_contexts, 
            context_dim = 9, 
            num_actions = 7, 
            noise_std = 0, 
            pi = 'eps-greedy', 
            eps = 0.1, 
            subset_r = 0.5,
            remove_underrepresented=False, 
        ):
        self.name = 'statlog'
        file_name = 'data/shuttle.trn'
        with open(file_name, 'r') as f:
            dataset = np.loadtxt(f)

        ## Meta-info 
        name = 'statlog'
        self.num_contexts = num_contexts 
        self.num_test_contexts = num_test_contexts
        self.context_dim = context_dim
        self.num_actions = num_actions
        self.pi = pi 
        self.eps = eps 
        self.subset_r = subset_r
        self.noise_std = noise_std


        contexts = dataset[: :-1]
        labels = dataset[:, -1].astype(int) - 1
        if remove_underrepresented:
            contexts, labels = remove_underrepresented_classes(contexts, labels)
        self.contexts, self.mean_rewards = classification_to_bandit_problem(contexts, labels, num_actions)

        reward_type = 'deterministic'
        print('{}: num_actions: {} | total_samples: {} | context_dim: {} | reward_type: {}'.format(
            name, num_actions, self.num_samples, context_dim, reward_type
        ))

    
    @property
    def num_samples(self):
        return self.contexts.shape[0]

    def reset_data(self, sim_id=0):
        # Load meta-data to generate dataset
        indices = np.load('data/meta/indices_{}.npy'.format(sim_id)) # random permutation of np.arange(100000)
        test_indices = np.load('data/meta/test_indices_{}.npy'.format(sim_id)) # random permutation of np.arange(100000)

        # Generate inds
        indices = indices % self.num_samples
        test_indices = test_indices % self.num_samples

        if self.num_contexts > self.num_samples:
            ind = indices[:self.num_contexts]
        else:
            # then select self.num_contexts first distinc elements of indices
            i = np.unique(indices,return_index=True)[1]
            i.sort()
            ind = indices[i[:self.num_contexts]]

        if self.num_test_contexts > self.num_samples:
            test_ind = test_indices[:self.num_test_contexts]
        else:
            i = np.unique(test_indices,return_index=True)[1]
            i.sort()
            test_ind = test_indices[i[:self.num_test_contexts]]

        contexts, mean_rewards = self.contexts[ind,:], self.mean_rewards[ind,:] 
        test_contexts, mean_test_rewards = self.contexts[test_ind,:], self.mean_rewards[test_ind,:]

        #create rewards
        rewards = mean_rewards + self.noise_std * np.random.normal(size=mean_rewards.shape)
        actions = sample_offline_policy(mean_rewards, self.num_contexts, self.num_actions, self.pi, self.eps, self.subset_r, contexts, rewards)

        dataset = (contexts, actions, rewards, test_contexts, mean_test_rewards) 
        return dataset 
# Classification of pixels into 7 forest cover types based on attributes 
# such as elevation, aspect, slope, hillshade, soil-type, and more.
class CoverTypeData(object): 
    def __init__(self,
            num_contexts, 
            num_test_contexts, 
            num_actions = 7, 
            context_dim = 54,
            noise_std = 0, 
            pi = 'eps-greedy', 
            eps = 0.1, 
            subset_r = 0.5,
            remove_underrepresented=False 
        ):
        self.name = 'covertype'
        file_name = 'data/covtype.data'
        with open(file_name, 'r') as f:
            df = pd.read_csv(f, header=0, na_values=['?']).dropna()

        name = 'covertype'
        self.num_contexts = num_contexts 
        self.num_test_contexts = num_test_contexts
        self.context_dim = context_dim
        self.num_actions = num_actions
        self.pi = pi 
        self.eps = eps 
        self.subset_r = subset_r
        self.noise_std = noise_std

            
        # data = df.iloc[:num_contexts, :]

        # Assuming what the paper calls response variable is the label?
        # Last column is label.
        labels = df[df.columns[-1]].astype('category').cat.codes.to_numpy()
        df = df.drop([df.columns[-1]], axis=1)

        if remove_underrepresented:
            df, labels = remove_underrepresented_classes(df, labels)

        contexts = df.to_numpy()
        self.contexts, self.rewards = classification_to_bandit_problem(contexts, labels, num_actions)
        
        reward_type = 'deterministic'
        print('{}: num_actions: {} | total_samples: {} | context_dim: {} | reward_type: {}'.format(
            name, num_actions, self.num_samples, context_dim, reward_type
        ))

    def reset_data(self, sim_id=0):
        # Load meta-data to generate dataset
        indices = np.load('data/meta/indices_{}.npy'.format(sim_id)) # random permutation of np.arange(100000)
        test_indices = np.load('data/meta/test_indices_{}.npy'.format(sim_id)) # random permutation of np.arange(100000)

        # Generate inds
        indices = indices % self.num_samples
        test_indices = test_indices % self.num_samples

        if self.num_contexts > self.num_samples:
            ind = indices[:self.num_contexts]
        else:
            # then select self.num_contexts first distinc elements of indices
            i = np.unique(indices,return_index=True)[1]
            i.sort()
            ind = indices[i[:self.num_contexts]]

        if self.num_test_contexts > self.num_samples:
            test_ind = test_indices[:self.num_test_contexts]
        else:
            i = np.unique(test_indices,return_index=True)[1]
            i.sort()
            test_ind = test_indices[i[:self.num_test_contexts]]


        contexts = self.contexts[ind,:] 
        mean_rewards = self.rewards[ind,:] 
        test_contexts = self.contexts[test_ind,:]
        mean_test_rewards = self.rewards[test_ind,:]
        actions = sample_offline_policy(mean_rewards, self.num_contexts, self.num_actions, self.pi, self.eps, self.subset_r)
        #create rewards
        rewards = mean_rewards + self.noise_std * np.random.normal(size=mean_rewards.shape)
        dataset = (contexts, actions, rewards, test_contexts, mean_test_rewards) 
        return dataset 

    @property 
    def num_samples(self):
        return self.contexts.shape[0]

class StockData(object):
    def __init__(self, 
                num_contexts, 
                num_test_contexts, 
                num_actions = 8, 
                noise_std=0.1, 
                pi = 'eps-greedy', 
                eps = 0.1, 
                subset_r = 0.5
                ): 
        self.name = 'stock'
        filename = 'data/raw_stock_contexts'
        self.num_contexts = num_contexts 
        self.num_test_contexts = num_test_contexts
        self.num_actions = num_actions 

        self.noise_std = noise_std
        self.pi = pi
        self.eps = eps
        self.subset_r = subset_r


        with open(filename, 'r') as f:
            contexts = np.loadtxt(f, skiprows=1)

        self.contexts = contexts
        self.context_dim = contexts.shape[1]

        print('Stock: num_samples: {}'.format(self.num_samples)) 

    @property 
    def num_samples(self): 
        return len(self.contexts)


    def reset_data(self, sim_id=0): 
        # Load meta-data to generate dataset
        indices = np.load('data/meta/indices_{}.npy'.format(sim_id)) # random permutation of np.arange(100000)
        test_indices = np.load('data/meta/test_indices_{}.npy'.format(sim_id)) # random permutation of np.arange(100000)
        self.betas = np.load('data/meta/betas_{}.npy'.format(sim_id))

        # Generate inds 
        indices = indices % self.num_samples
        test_indices = test_indices % self.num_samples

        if self.num_contexts > self.num_samples:
            ind = indices[:self.num_contexts]
        else:
            # then select self.num_contexts first distinc elements of indices
            i = np.unique(indices,return_index=True)[1]
            i.sort()
            ind = indices[i[:self.num_contexts]]
        
        if self.num_test_contexts > self.num_samples:
            test_ind = test_indices[:self.num_test_contexts]
        else:
            i = np.unique(test_indices,return_index=True)[1]
            i.sort()
            test_ind = test_indices[i[:self.num_test_contexts]]

        contexts = self.contexts[ind,:] 
        test_contexts = self.contexts[test_ind,:]
        # Compute rewards
        mean_rewards = np.dot(contexts, self.betas) # (num_contexts, num_actions)
        mean_test_rewards = np.dot(test_contexts, self.betas) # (num_contexts, num_actions)

        actions = sample_offline_policy(mean_rewards, self.num_contexts, self.num_actions, self.pi, self.eps, self.subset_r)
        #create rewards
        rewards = mean_rewards + self.noise_std * np.random.normal(size=mean_rewards.shape)
        dataset = (contexts, actions, rewards, test_contexts, mean_test_rewards) 
        return dataset 

## seems like adultdata is most similar to sepsis data
class AdultData(object):
    def __init__(self, num_contexts, 
            num_test_contexts, 
            num_actions = 14, 
            noise_std=0., 
            remove_underrepresented=False,
            pi = 'eps-greedy', 
            eps = 0.1, 
            subset_r = 0.5
            ): 
        self.name = 'adult'
        self.num_contexts = num_contexts 
        self.num_test_contexts = num_test_contexts
        self.num_actions = num_actions 

        self.noise_std = noise_std
        self.pi = pi
        self.eps = eps
        self.subset_r = subset_r


        file_name = 'data/adult.data'
        with open(file_name, 'r') as f:
            df = pd.read_csv(f, header=None, na_values=[' ?']).dropna()
        # print(f'df.columns: {df.columns}')

        labels = df[6].astype('category').cat.codes.to_numpy()
        df = df.drop([6], axis=1)

        # Convert categorical variables to 1 hot encoding
        cols_to_transform = [1, 3, 5, 7, 8, 9, 13, 14]
        df = pd.get_dummies(df, columns=cols_to_transform)

        if remove_underrepresented:
            df, labels = remove_underrepresented_classes(df, labels)
        contexts = df.to_numpy()
        self.context_dim = contexts.shape[1]
        print(f'contexts.type:{contexts.shape}')

        self.contexts, self.rewards = classification_to_bandit_problem(contexts, labels, num_actions)
        print('Adult: num_samples: {}'.format(self.contexts.shape[0]))

    @property
    def num_samples(self):
        return self.contexts.shape[0]
    
    def reset_data(self, sim_id=0):
        # Load meta-data to generate dataset
        indices = np.load('data/meta/indices_{}.npy'.format(sim_id)) # random permutation of np.arange(100000)
        test_indices = np.load('data/meta/test_indices_{}.npy'.format(sim_id)) # random permutation of np.arange(100000)

        # Generate inds
        indices = indices % self.num_samples
        test_indices = test_indices % self.num_samples

        if self.num_contexts > self.num_samples:
            ind = indices[:self.num_contexts]
        else:
            # then select self.num_contexts first distinc elements of indices
            i = np.unique(indices,return_index=True)[1]
            i.sort()
            ind = indices[i[:self.num_contexts]]

        if self.num_test_contexts > self.num_samples:
            test_ind = test_indices[:self.num_test_contexts]
        else:
            i = np.unique(test_indices,return_index=True)[1]
            i.sort()
            test_ind = test_indices[i[:self.num_test_contexts]]


        contexts = self.contexts[ind,:] 
        mean_rewards = self.rewards[ind,:] 
        test_contexts = self.contexts[test_ind,:]
        mean_test_rewards = self.rewards[test_ind,:]
        #create rewards
        rewards = mean_rewards + self.noise_std * np.random.normal(size=mean_rewards.shape)
        actions = sample_offline_policy(mean_rewards, self.num_contexts, self.num_actions, self.pi, self.eps, self.subset_r, contexts, rewards)

        dataset = (contexts, actions, rewards, test_contexts, mean_test_rewards) 
        return dataset 


class CensusData(object):
    def __init__(self, num_contexts, 
            num_test_contexts, 
            num_actions = 9, 
            noise_std=0., 
            remove_underrepresented=False,
            pi = 'eps-greedy', 
            eps = 0.1, 
            subset_r = 0.5
            ): 
        self.name = 'census'
        self.num_contexts = num_contexts 
        self.num_test_contexts = num_test_contexts
        self.num_actions = num_actions 

        self.noise_std = noise_std
        self.pi = pi
        self.eps = eps
        self.subset_r = subset_r

        file_name = 'data/USCensus1990.data.txt'
        with open(file_name, 'r') as f:
            df = (pd.read_csv(f, header=0, na_values=['?']).dropna())

        # Assuming what the paper calls response variable is the label?
        labels = df['dOccup'].astype('category').cat.codes.to_numpy()
        # print(labels)
        # In addition to label, also drop the (unique?) key.
        df = df.drop(['dOccup', 'caseid'], axis=1)

        # All columns are categorical. Convert to 1 hot encoding.
        df = pd.get_dummies(df, columns=df.columns)

        if remove_underrepresented:
            df, labels = remove_underrepresented_classes(df, labels)
        contexts = df.to_numpy()
        self.contexts, self.rewards = classification_to_bandit_problem(contexts, labels, num_actions)
        self.context_dim = self.contexts.shape[1] # 389
        
        print('Census: num_samples: {}'.format(self.contexts.shape[0]))

    @property
    def num_samples(self):
        return self.contexts.shape[0]
    
    def reset_data(self, sim_id=0):
        # Load meta-data to generate dataset
        indices = np.load('data/meta/indices_{}.npy'.format(sim_id)) # random permutation of np.arange(100000)
        test_indices = np.load('data/meta/test_indices_{}.npy'.format(sim_id)) # random permutation of np.arange(100000)

        # Generate inds
        indices = indices % self.num_samples
        test_indices = test_indices % self.num_samples

        if self.num_contexts > self.num_samples:
            ind = indices[:self.num_contexts]
        else:
            # then select self.num_contexts first distinc elements of indices
            i = np.unique(indices,return_index=True)[1]
            i.sort()
            ind = indices[i[:self.num_contexts]]

        if self.num_test_contexts > self.num_samples:
            test_ind = test_indices[:self.num_test_contexts]
        else:
            i = np.unique(test_indices,return_index=True)[1]
            i.sort()
            test_ind = test_indices[i[:self.num_test_contexts]]


        contexts = self.contexts[ind,:] 
        mean_rewards = self.rewards[ind,:] 
        test_contexts = self.contexts[test_ind,:]
        mean_test_rewards = self.rewards[test_ind,:]
        actions = sample_offline_policy(mean_rewards, self.num_contexts, self.num_actions, self.pi, self.eps, self.subset_r)
        #create rewards
        rewards = mean_rewards + self.noise_std * np.random.normal(size=mean_rewards.shape)
        dataset = (contexts, actions, rewards, test_contexts, mean_test_rewards) 
        return dataset 
    
def safe_std(values):
    """
    Remove zero std values for ones.
    using a standard deviation of zero in normalization or scaling operations 
    can lead to division by zero errors or undefined behavior. 
    """
    values = np.array(values)
    values[values==0]=1.0
    return values
    # return np.array([val if val != 0.0 else 1.0 for val in values])
    # return np.array([val if val != 0 else 1.0 for val in values])



def classification_to_bandit_problem(contexts, labels, num_actions=None):
    """Normalize contexts and encode deterministic rewards."""
    # is not actions is specified, the action is the prediction class
    if num_actions is None:
        num_actions = np.max(labels) + 1
    num_contexts = contexts.shape[0]

    # Due to random subsampling in small problems, some features may be constant
    # ? if the features are constant, the std = 0, replace them with 1
    sstd = safe_std(np.std(contexts, axis=0, keepdims=True)[0, :])

    # Normalize features to have zero mean and unit variance
    contexts = ((contexts - np.mean(contexts, axis=0, keepdims=True)) / sstd)

    # One hot encode labels as rewards
    '''
    each row corresponds to a context, and each column corresponds to an action
    initialize the rewards to zeros
    selecting the correct action (class) yields reward 1, otherwise 0
    '''
    rewards = np.zeros((num_contexts, num_actions))
    rewards[np.arange(num_contexts), labels] = 1.0

    return contexts, rewards #, (np.ones(num_contexts), labels)

  

def remove_underrepresented_classes(features, labels, thresh=0.0005):
    """Removes classes when number of datapoints fraction is below a threshold."""

    # Threshold doesn't seem to agree with https://arxiv.org/pdf/1706.04687.pdf
    # Example: for Covertype, they report 4 classes after filtering, we get 7?
    total_count = labels.shape[0]
    unique, counts = np.unique(labels, return_counts=True)
    ratios = counts.astype('float') / total_count
    vals_and_ratios = dict(zip(unique, ratios))
    print('Unique classes and their ratio of total: %s' % vals_and_ratios)
    keep = [vals_and_ratios[v] >= thresh for v in labels]
    return features[keep], labels[np.array(keep)]

from ucimlrepo import fetch_ucirepo
if __name__ == '__main__':
     
    # Generate meta data, do not run it
    print('WARNING: This is to generate meta data for dataset generation, and should only be performed once. Quit now if you are not sure what you are doing!!!')
    s = input('Type yesimnotstupid to proceed: ')
    if s == 'yesimnotstupid':
        for sim_id in range(10):
            np.random.seed(sim_id)
            # there will not be repeated numbers
            indices = np.random.permutation(100000)
            np.save('data/meta/indices_{}.npy'.format(sim_id), indices)

            np.random.seed(sim_id)
            test_indices = np.random.permutation(100000)
            np.save('data/meta/test_indices_{}.npy'.format(sim_id), test_indices)


            ### not sure what betas are doing...????
            np.random.seed(sim_id)
            context_dim = 21
            num_actions = 8
            # betas are uniformly distributed over the half-open interval [-1, 1)
            # size: context_dim X num_actions
            betas = np.random.uniform(-1, 1, (context_dim, num_actions))
            betas /= np.linalg.norm(betas, axis=0)
            np.save('data/meta/betas_{}.npy'.format(sim_id), betas)


            # fetch dataset 
            mushroom = fetch_ucirepo(id=73) 
            # data (as pandas dataframes) 
            X = mushroom.data.features 
            y = mushroom.data.targets 
            # print(f'mushroom.metadata')
            # metadata 
            # print(mushroom.metadata) 
            print(f'mushroom.variables')
            # variable information 
            print(mushroom.variables) 
            print(f'x: {X}')
            print(f'y: {y}')
    


# file_name = 'mushroom.data'
# contexts, rewards, off_action, exp_rewards = sample_mushroom_data(file_name, num_contexts=10000, p_opt=1, p_uni=0)

# file_name = 'shuttle.trn'
# contexts, rewards, off_action, exp_rewards = sample_statlog_data(file_name, num_contexts=10000, p_opt=1, p_uni=0)

# file_name = 'adult.data'
# contexts, rewards, off_action, exp_rewards = sample_adult_data(file_name, num_contexts=30, p_opt=1, p_uni=0)

# file_name = 'raw_stock_contexts'
# contexts, rewards, off_action, exp_rewards = sample_stock_data(file_name, 
#     num_contexts=10000, p_opt=1, p_uni=0)

# file_name = 'jester.npy'
# contexts, rewards, off_action, exp_rewards = sample_jester_data(file_name, 
#     num_contexts=10000, p_opt=1, p_uni=0)

# file_name = 'USCensus1990.data.txt'
# contexts, rewards, off_action, exp_rewards = sample_census_data(file_name, 
#     num_contexts=10000, p_opt=1, p_uni=0)

# file_name = 'covtype.data'
# contexts, rewards, off_action, exp_rewards = sample_covertype_data(file_name, 
#     num_contexts=10000, p_opt=1, p_uni=0)



# print(contexts.shape) 
# print(rewards.shape)
# print(exp_rewards.shape)
# print(off_action.shape)
# print(contexts)
