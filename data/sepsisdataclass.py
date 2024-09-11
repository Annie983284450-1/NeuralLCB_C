import sys
import os 

if os.name == 'nt':
    # windows
    neruallcb_path = r'E:\phd thesis ideas\NeuralLCB_C'
elif os.name == 'posix':
    if sys.platform == 'darwin':
        # neruallcb_path = r'C:\Users\98328\Desktop\phd thesis ideas\offline_neural_bandits-main'
        # macOS
        neruallcb_path = '/Users/anniezhou/Desktop/NeuralLCB_C'

if neruallcb_path not in sys.path:
    sys.path.append(neruallcb_path)
import numpy as np 
import pandas as pd 
from core.utils import sample_offline_policy
# import tensorflow as tf

def one_hot(df, cols):
    """Returns one-hot encoding of DataFrame df including columns in cols."""
    for col in cols:
        dummies = pd.get_dummies(df[col], prefix=col, drop_first=False)
        df = pd.concat([df, dummies], axis=1)
        df = df.drop(col, axis=1)
    return df
## analyze this one as example


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



def classification_to_bandit_problem_sepsis(contexts, labels, df,num_actions=None):
    """Normalize contexts and encode deterministic rewards."""
    # is not actions is specified, the action is the prediction class

    # Normalize only the continuous features
    contexts = df.copy()
    
    # Separate categorical and continuous features
    categorical_cols = ['Gender', 'Unit1', 'Unit2', 'ICULOS']  # columns to exclude from normalization, ICULOS specially included
    continuous_cols = [col for col in df.columns if col not in categorical_cols]
    # contexts[continuous_cols] = (df[continuous_cols] - df[continuous_cols].mean()) / df[continuous_cols].std()
    
    if num_actions is None:
        num_actions = np.max(labels) + 1
    num_contexts = contexts.shape[0]

    # Due to random subsampling in small problems, some features may be constant
    # ? if the features are constant, the std = 0, replace them with 1
    # sstd = safe_std(np.std(contexts, axis=0, keepdims=True)[0, :])
    sstd = safe_std(np.std(contexts[continuous_cols], axis=0, keepdims=True)[0, :])
    # Normalize features to have zero mean and unit variance
    contexts[continuous_cols] = ((contexts[continuous_cols] - np.mean(contexts[continuous_cols], axis=0, keepdims=True)) / sstd)

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

import numpy as np
import pandas as pd
'''
If rewards are deterministic (clear cause-effect relationship): You may not need any noise, and you can set noise_std to 0. 
If rewards are stochastic (uncertainty in rewards): 
In cases where the outcome (reward) has some inherent randomness or uncertainty 
(e.g., real-world situations with incomplete information), adding noise can better simulate the real-world conditions. 
In such cases, you could choose a small positive value for noise_std.
start from 0.01 to 0.1 for reward is either 0 or 1 (small values)
'''
class SepsisData(object):
    def __init__(self, file_name, 
                 train_patients_file, 
                 test_patients_file, 
                 num_actions=2, 
                 noise_std=0.,
                pi='eps-greedy', 
                eps=0.1, 
                subset_r=0.5):
        """
        Initialize the Sepsis dataset for contextual bandit setting.
        :param file_name: path to the sepsis dataset CSV file.
        :param train_patients_file: path to the numpy file containing patient IDs for training.
        :param test_patients_file: path to the numpy file containing patient IDs for testing.
        :param num_actions: number of actions (for binary sepsis classification, 2 actions: sepsis and non-sepsis).
        :param noise_std: noise to add to the rewards (set to 0 since you don't need noise).
        :param pi: policy to use ('eps-greedy' in this case).
        :param eps: epsilon value for epsilon-greedy policy.
        :param subset_r: subset ratio for selecting part of the data.
        """
        self.num_actions = num_actions
        self.noise_std = noise_std
        self.pi = pi
        self.eps = eps
        self.subset_r = subset_r
        train_patients_file = './data/SepsisData/train_set.npy'
        test_patients_file = './data/SepsisData/test_set.npy'
        # Load train and test patient IDs
        train_patients = np.load(train_patients_file, allow_pickle=True)
        test_patients = np.load(test_patients_file, allow_pickle=True)
        file_name = './data/SepsisData/fully_imputed.csv'
        # Load dataset
        df = pd.read_csv(file_name)

        # Drop the column 'HospAdmTim' as per your requirement
        df = df.drop(['HospAdmTim'], axis=1)
        
        # Separate dataset into train and test sets based on patient IDs
        train_df = df[df['pat_id'].isin(train_patients)]
        test_df = df[df['pat_id'].isin(test_patients)]
        
        # Extract labels (SepsisLabel) and drop it from the feature set
        train_labels = train_df['SepsisLabel'].to_numpy()
        test_labels = test_df['SepsisLabel'].to_numpy()

        # Drop unnecessary columns
        train_df = train_df.drop(['SepsisLabel', 'pat_id','hours2sepsis'], axis=1)
        test_df = test_df.drop(['SepsisLabel', 'pat_id', 'hours2sepsis'], axis=1)
        
        # Convert the remaining columns (features) to numpy arrays
        train_contexts = train_df.to_numpy()
        test_contexts = test_df.to_numpy()
        
        # Convert the classification problem into a contextual bandit problem
        self.train_contexts, self.train_rewards = classification_to_bandit_problem_sepsis(train_contexts, train_labels,train_df,  num_actions)
        self.test_contexts, self.test_rewards = classification_to_bandit_problem_sepsis(test_contexts, test_labels, test_df, num_actions)
        
        # Set the context dimension
        self.context_dim = self.train_contexts.shape[1]  # Number of features

        print(f'SepsisData: train_samples: {self.num_train_samples}, test_samples: {self.num_test_samples}')
    
    @property
    def num_train_samples(self):
        return self.train_contexts.shape[0]
    
    @property
    def num_test_samples(self):
        return self.test_contexts.shape[0]

    def reset_data(self, sim_id=0):
        """
        Reset the dataset using random indices for simulation purposes.
        """
        # No shuffling or leakage is allowed since we are grouping by patient ID
        train_indices = np.arange(self.num_train_samples)
        test_indices = np.arange(self.num_test_samples)
        
        contexts = self.train_contexts[train_indices, :]
        mean_rewards = self.train_rewards[train_indices, :]
        test_contexts = self.test_contexts[test_indices, :]
        mean_test_rewards = self.test_rewards[test_indices, :]
        
        # Generate rewards (if needed, noise can be added here)
        rewards = mean_rewards + self.noise_std * np.random.normal(size=mean_rewards.shape)
        
        # Simulate actions based on epsilon-greedy policy
        actions = sample_offline_policy(mean_rewards, self.num_train_samples, self.num_actions, self.pi, self.eps, self.subset_r, contexts, rewards)
        
        dataset = (contexts, actions, rewards, test_contexts, mean_test_rewards)
        return dataset
    


 
# class SepsisData(object):
#     def __init__(self,
#                  num_contexts,
#                  num_test_contexts,
#                 #  context_dim = 13,
#                  num_actions = 2, # septic or non septic
#                  noise_std = 0,
#                  pi = 'eps-greedy', 
#                  eps = 0.1, 
#                  subset_r = 0.5,
#                 #  remove_underrepresented=True,
#                  ):
#         self.name = 'sepsis'
#         # file_name = 'data/SepsisData/fully_imputed_8windowed_max48_updated.csv'
#         # with open(file_name,'r') as f:
#         #     df = pd.read_csv(f, header = 0)
#         # sepsis_full = pd.read_csv(f'./data/SepsisData/fully_imputed_8windowed_max48_updated.csv')
#         sepsis_full = pd.read_csv(f'./data/SepsisData/fully_imputed.csv')
#         sepsis_full = sepsis_full.drop(['HospAdmTim', 'hours2sepsis'])
#         self.num_contexts = num_contexts 
#         self.num_test_contexts = num_test_contexts
#         self.num_actions = num_actions
#         self.pi = pi 
#         self.eps = eps  
#         self.subset_r = subset_r
#         self.noise_std = noise_std


#         sepsis_train_wins = np.load('./data/SepsisData/sepsis_train_wins.npy')
#         sepsis_train_wins = sepsis_train_wins.tolist()
#         nosepsis_train_wins = np.load('./data/SepsisData/nosepsis_train_wins.npy')
#         nosepsis_train_wins = nosepsis_train_wins.tolist()

#         train_wins = sepsis_train_wins[0:int(self.num_contexts/3)]+nosepsis_train_wins[0:int(self.num_contexts/3*2)]    
#         # print(f'train_wins:{train_wins}')
#         print(f'num_contexts: {self.num_contexts}')
#         print(f'length of train wins:{len(train_wins)}')

#         test_septic_wins = np.load('./Data/test_septic_wins.npy')
#         test_septic_wins = test_septic_wins.tolist()
#         test_noseptic_wins = np.load('./Data/test_noseptic_wins.npy')
#         test_noseptic_wins = test_noseptic_wins.tolist()
#         # num_test_pat_noseptic_win = math.floor(self.num_test_pat_win*12)
#         test_wins = test_septic_wins[0:int(self.num_test_contexts/2)]+test_noseptic_wins[0:int(self.num_test_contexts/2)]
#         # print(f'test_wins:{test_wins}')
#         print(f'num of test contexts: {self.num_test_contexts}')
#         print(f'length of test wins:{len(test_wins)}')

         
#         train_df = sepsis_full[sepsis_full['pat_id'].isin(train_wins)]
#         labels = train_df['SepsisLabel'].to_numpy()

       
#         train_df = train_df.drop(columns=['SepsisLabel','hours2sepsis','HospAdmTime', 'pat_id'])
#         contexts = train_df.to_numpy() 


#         test_df = sepsis_full[sepsis_full['pat_id'].isin(test_wins)]
#         test_labels = test_df['SepsisLabel'].to_numpy()  
#         test_df = test_df.drop(columns=['SepsisLabel','hours2sepsis','HospAdmTime', 'pat_id'])
#         test_contexts= test_df.to_numpy() 


#         """Normalize contexts and encode deterministic rewards."""
#         self.contexts, self.mean_rewards = classification_to_bandit_problem(contexts, labels, num_actions)
#         print(f'contexts for sepsis:{self.contexts}')
#         print(f'mean rewards for sepsis:{self.mean_rewards}')
#         self.test_contexts, self.test_mean_rewards = classification_to_bandit_problem(test_contexts, test_labels, num_actions)

#         self.context_dim = contexts.shape[1]


#     @property
#     def num_samples(self):
#         return self.contexts.shape[0]


#     @property 
#     def num_test_samples(self):
#         return self.test_contexts.shape[0]
    


#     def reset_data(self, sim_id = 0):
#         contexts = self.contexts 
#         mean_rewards = self.mean_rewards 
#         test_contexts = self.test_contexts 
#         mean_test_rewards = self.test_mean_rewards

         
        
#         actions = sample_offline_policy(mean_rewards, self.num_samples, self.num_actions, self.pi, self.eps, self.subset_r)
#         #create rewards 
#         # based on the paper, the gaussian noise is a standard in stochastic bandit literature
#         rewards = mean_rewards + self.noise_std * np.random.normal(size=mean_rewards.shape)
#         dataset = (contexts, actions, rewards, test_contexts, mean_test_rewards) 
        
#         return dataset 