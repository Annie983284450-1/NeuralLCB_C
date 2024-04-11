"""Define a uniform sampling algorithm. """

import jax.numpy as jnp
import optax
import numpy as np 
from core.bandit_algorithm import BanditAlgorithm 
from core.bandit_dataset import BanditDataset

# subclass of BanditAlgorithm
'''
he UniformSampling algorithm does not use the information contained in the contexts to make decisions. 
Instead, it selects actions with equal probability across all possible actions for each decision point. 
This strategy is often used as a baseline in bandit experiments to evaluate the performance of more sophisticated algorithms 
that aim to learn and adapt based on observed rewards and contexts.
'''
class UniformSampling(BanditAlgorithm):
    def __init__(self, hparams, update_freq=1, name='Uniform'):
        self.name = name 
        self.hparams = hparams
        self.update_freq = update_freq 
# takes a batch of contexts as input and returns a batch of actions chosen according to the uniform sampling strategy
    def sample_action(self, contexts):
        return np.random.randint(0, self.hparams.num_actions, (contexts.shape[0],)) 