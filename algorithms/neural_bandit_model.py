"""Define  neural network models for (contextual) bandits.  

This model takes a context as input and predicts the rewards of all actions:

                            f(x,a) = < phi(x), w_a > 
"""

import numpy as np 
from core.nn import NeuralNetwork 
from core.utils import action_convolution 
import haiku as hk
import jax 
import jax.numpy as jnp
import optax
from tqdm import tqdm
import sys
import copy
"""
class NeuralNetwork is A neural network prototype with 3 basic functions:
__init__()
build_model()
train()
""" 

'''
Difference between NeuralBanditModel and NeuralBanditModelV2:

    NeuralBanditModel outputs a vector of predicted rewards for all actions, one for each action. 
    The action is selected by the algorithm after the forward pass based on these predicted rewards.
'''


'''

Aspect	           |           NeuralBanditModel	        |           NeuralBanditModelV2
Action Handling    |	Predicts rewards for all actions	|   Predicts reward for a specific action
Forward Pass       |	Action-independent	                |   Action-dependent
Action Convolution |    No	                                |   Yes

'''


class NeuralBanditModel(NeuralNetwork):
    """Build a neural network model for bandits.
     
    This model takes a context as input and predict the expected rewards of multiple actions. 
    The NeuralBanditModelV2 takes the context and the action together as input. 
    The context is convolved with the action using the action_convolution_impure_fn, which effectively conditions the context on the specific action.
    """

    def __init__(self, optimizer, hparams, name):
        self.optimizer = optimizer 
        self.hparams = hparams 
        self.name = name 
        # Using the smallest layer size helps in stabilizing the training process because it prevents overestimating the model’s capacity based on larger layers
        self.m = min(self.hparams.layer_sizes)
        self.build_model()
        print('{} has {} parameters.'.format(name, self.num_params))
    def clone(self):
        return copy.deepcopy(self)
    
    def reset(self, seed):
        self.init(seed)
    def build_model(self):
        """Transform impure functions into pure functions and apply JAX tranformations."""
        # hk.transform transform the nn to sth JAX can process
        '''
        hk.without_apply_rng is a utility function in Haiku that modifies the behavior of the apply function returned by hk.transform.
        By default, the apply function expects a random number generator (RNG) key as one of its arguments, 
        which is used for operations that require randomness, such as dropout. 
        If your model doesn't use any such stochastic operations during the apply phase, 
        you can use hk.without_apply_rng to create a version of the apply function that doesn't require an RNG key as an input.
        '''
        # transform(f, [,apply_rng]) trandform a function using Haiku modules into a pair of pure functions
        # without_apply_rng() remove the rng argument from the apply function, See out_impure_fn(self, params, context), where nn is apply()
        
        self.nn = hk.without_apply_rng(hk.transform(self.net_impure_fn))
        # When you apply jax.jit to a function, JAX traces the function, captures the operations performed in it, 
        # and compiles them into an optimized sequence of instructions that can run on GPU or TPU
        # out_impure_fn: return self.nn.apply(params, context)
        # def out_impure_fn(self, params, contexts, actions):
        #     return self.nn.apply(params, contexts, actions)
        self.out = jax.jit(self.out_impure_fn) 
        self.grad_out = jax.jit(self.grad_out_impure_fn)
        self.loss = jax.jit(self.loss_impure_fn) 
        self.update = jax.jit(self.update_impure_fn)   
        
        # Initialize network parameters and opt states. 
        """   
         The model's parameters (self.params) are initialized after the above compilation steps 
         because JAX compiles the abstract structure of operations rather than their specific instances. 
         The init method's purpose is to create and store the initial parameters, which can then be used with the compiled functions. 
         This approach separates the concerns of defining and optimizing the model's computational graph from the instantiation of 
         model parameters, following JAX's functional programming paradigm.
        """            
        self.init()             
   
        # Compute number of network parameters 
        p = (self.hparams.context_dim + 1) * self.hparams.layer_sizes[0] 
        for i in range(1, len(self.hparams.layer_sizes)):
            p += (self.hparams.layer_sizes[i-1] + 1) * self.hparams.layer_sizes[i]
        p += self.hparams.layer_sizes[-1] + 1         
        if self.hparams.layer_n: 
            p += sum(2 * self.hparams.layer_sizes)
        self.num_params = p 

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

    # defines the structure of the neural network
    def net_impure_fn(self, context):
        """
        Args:
            context: context, (None, self.hparams.context_dim)
        """
        net_structure = []
        # 'layer_sizes': [100,100], 
        # each integer specifies the number of units in a layer
        for num_units in self.hparams.layer_sizes:
            # for each layer, create a linear (fully connected) layer with hk.Linear
            # 
            net_structure.append(  
                hk.Linear(
                    # hk.initializers.UniformScaling,  scales the weights uniformly. 
                    num_units, w_init=hk.initializers.UniformScaling(self.hparams.s_init) 
                    )
                ) 
            if self.hparams.layer_n: 
                # hk.LayerNorm(): applies layer normalization to input
                # Layer normalization is a technique to normalize the inputs across the features for each layer, 
                # which can help stabilize and speed up training. 
                # axis=1 specifies that normalization is applied across each feature.
                net_structure.append(hk.LayerNorm(axis=1, create_scale=True, create_offset=True))
            # activation function: jax.nn.relu
            net_structure.append(self.hparams.activation) 
        # output layer. Predicting the reward for each action
        net_structure.append(
                hk.Linear(self.hparams.num_actions, w_init=hk.initializers.UniformScaling(self.hparams.s_init) )
            )
        # class haiku.Sequential(layers, name=None)
        mlp = hk.Sequential(net_structure) 

        return mlp(context)
    
    def out_impure_fn(self, params, context):
        # execute the neural network forward pass
        out_network = self.nn.apply(params, context)
        # print(f'type(out_network):{out_network}')
        # print(f'out_network:{out_network}')
        return out_network
        # return self.nn.apply(params, context)
 
    def grad_out_impure_fn(self, params, context):
        """
        Args:
            params: Network parameters 
            context: (num_samples, context_dim)

        Return:
            action_grad_params: (num_actions, num_samples, p)
        """
        # self.out = jax.jit(self.out_impure_fn) 
        # compute the Jacobian of self.out
        '''
        jax.jacrev(self.out)(params, context): 
        This line computes the Jacobian of the self.out function with respect to its first argument (params), 
        evaluated at the points params and context. 
        The self.out function is expected to be the neural network's forward pass, which takes params and context as inputs. 
        jax.jacrev calculates the gradient of the output with respect to params for each element in the context.
        grad_params: This is the computed gradient (Jacobian) of the model's outputs with respect to its parameters.

        If a neural network outputs a vector of predictions based on a vector of inputs, 
        the Jacobian matrix of the network's output function with respect to its inputs provides detailed information on 
        how each input dimension influences each output dimension.
        '''
        grad_params = jax.jacrev(self.out)(params, context)
        # return grad_params

        action_grad_params = []

        '''
        grad_param_leaves = jax.tree_leaves(grad_params): 
        This line extracts all the leaf nodes (individual gradient tensors) from the nested grad_params structure. 
        JAX represents gradients as pytrees (nested Python containers of arrays), and jax.tree_leaves() flattens this structure into a list of arrays.
        '''
        grad_param_leaves = jax.tree_leaves(grad_params)

        # print(f'grad_param_leaves: {grad_param_leaves}')
        # print(f'grad_param_leaves.shape: {grad_param_leaves.shape}')


        # process the gradient info action by action
        for a in range(self.hparams.num_actions):
            # initialize a list to store the gradients relevant to current action a
            action_grad_param = []
            """
            for grad_param in grad_param_leaves[:-2]: iterates over all gradient tensors stored in grad_param_leaves, 
            excluding the last two. These tensors represent the gradients of the network's parameters for all actions, 
            and the loop processes each to isolate the part relevant to the current action a.
            """
            for grad_param in grad_param_leaves[:-2]:
                grad_param = grad_param.reshape((context.shape[0], self.hparams.num_actions, -1)) 
                action_grad_param.append( grad_param[:,a,:] ) 
            
            # for the last layer
            grad_param = grad_param_leaves[-2]

            # print(f'grad_param: {grad_param} for action {a}')
            # print(f'grad_param.shape: {grad_param.shape}')

            # ????????????  why grad_param[:,a,a]???
            action_grad_param.append( grad_param[:,a,a].reshape(-1,1)) 

            grad_param = grad_param_leaves[-1]
            action_grad_param.append(grad_param[:,a,:,a]) 

            action_grad_param = jnp.hstack(action_grad_param)
            action_grad_params.append(action_grad_param)

        return jnp.array(action_grad_params)
    '''
    loss_impure_fn:
    calculte the loss for a neural network model given parameters contexts, action and reward. 
    predictions is computed through self.nn
    then calculates the squared loss between these predictions and actual rewards, 
    weighted by the action vectors. 
    Additionally, it includes a regularization loss computed as the sum of squares of all model parameters, 
    scaled by a regularization parameter lambd. 
    The total loss is the sum of the squared and regularization losses. 
    This function is crucial for training the model, 
    guiding parameter updates to minimize prediction error and prevent overfitting through regularization.
    '''

    def loss_impure_fn(self, params, context, action, reward):
        """
        Args:
            context: An array of context, (None, self.hparams.context_dim)
            action: An array of one-hot action vectors, 1 for selected action and 0 other wise, 
            (None, self.hparams.num_actions)
            reward: An array of reward vectors, (None, self.hparams.num_actions)
        """
        # self.out = jax.jit(self.out_impure_fn) 
        preds = self.out(params, context) 

        # Batch size B is implicitly handled by the averaging function jnp.mean(), which divides the total error by the batch size.
        squared_loss = 0.5 * jnp.mean(jnp.sum(action * jnp.square(preds - reward), axis=1), axis=0)
        reg_loss = 0.5 * self.hparams.lambd * sum(
                jnp.sum(jnp.square(param)) for param in jax.tree_leaves(params) 
            )

        return squared_loss + reg_loss 
    
 

    def update_impure_fn(self, params, opt_state, context, action, reward):
        """
        Args:
            context: An array of context, (None, self.hparams.context_dim)
            action: An array of one-hot action vectors, 1 for selected action and 0 other wise, (None, self.hparams.num_actions)
            reward: An array of reward vectors, (None, self.hparams.num_actions)
        """
        grads = jax.grad(self.loss)(params, context, action, reward)
        updates, opt_state = self.optimizer.update(grads, opt_state)
        new_params = optax.apply_updates(params, updates)
        return new_params, opt_state

    def init(self): 
        key = jax.random.PRNGKey(self.hparams.seed)
        key, subkey = jax.random.split(key)

        context = jax.random.normal(key, (1, self.hparams.context_dim))

        '''
        use Haiku's init method for the neural network,
        which requires a sample input to determine the shapes of all the parameters. 
        A random context is generated for this purpose, 
        and then self.nn.init is called with this context to initialize params. 
        The optimizer state is also initialized based on these parameters. 
        This process sets up the model for training or inference by establishing initial values for all learnable parameters.
        '''
        self.params = self.nn.init(subkey, context) 
        self.opt_state = self.optimizer.init(self.params)

    def train(self, data, num_steps):
        if self.hparams.verbose:
            print('Training {} for {} steps.'.format(self.name, num_steps)) 
            
        params, opt_state = self.params, self.opt_state 
        for step in range(num_steps):
            x,w,y = data.get_batch_with_weights(self.hparams.batch_size) #(None,d), (None, num_actions), (None, num_actions)
            # print('DEBUG', x.shape, w.shape, y.shape)
            params, opt_state = self.update(params, opt_state, x,w,y) 

            if step % self.hparams.freq_summary == 0 and self.hparams.verbose:
                cost = self.loss(params, x,w,y)
                print('{} | step: {} | loss: {}'.format(self.name, step, cost))
        
        self.params, self.opt_state = params, opt_state




# sub class of NeuralBanditModel
class NeuralBanditModelV2(NeuralBanditModel):
    """Build a neural network model V2 for bandits.
    
    This model takes an action-convoluted context as input and predict the expected reward of the context. 
    """

    def __init__(self, optimizer, hparams, name='NeuralBanditModelV2' ):
        self.optimizer = optimizer 
        self.hparams = hparams 
        self.name = name 
        self.m = min(self.hparams.layer_sizes)
        self.build_model()
        # print('{} has {} parameters.'.format(name, self.num_params))
       
    def build_model(self):
        """Transform impure functions into pure functions and apply JAX tranformations."""
        #hk.transform will return 2 separate functions: init() and apply()

        # print(f'%%%%%%%%%%%%. Running NeuralBanditModelV2.build_model(). %%%%%%%%%%%%.')

        self.nn = hk.without_apply_rng(hk.transform(self.net_impure_fn))
        # self.out returns the predictions of the neural networks
        self.out = jax.jit(self.out_impure_fn) 
        self.grad_out = jax.jit(self.grad_out_impure_fn)
        # self.grad_out_cp = jax.jit(self.grad_out_impure_cp)
        # added action convolution???
        self.action_convolution = jax.jit(self.action_convolution_impure_fn)
        self.loss = jax.jit(self.loss_impure_fn)
       
        self.update = jax.jit(self.update_impure_fn)

        # Initialize network parameters and opt states. 
        
        self.init(self.hparams.seed)

        # Compute number of network parameters 
        p = (self.hparams.context_dim + 1) * self.hparams.layer_sizes[0] 
        for i in range(1, len(self.hparams.layer_sizes)):
            p += (self.hparams.layer_sizes[i-1] + 1) * self.hparams.layer_sizes[i]
        p += self.hparams.layer_sizes[-1] + 1 
        if self.hparams.layer_n: 
            p += sum(2 * self.hparams.layer_sizes)
        self.num_params = p 

    def reset(self, seed):
        self.init(seed)

    def net_impure_fn(self, contexts, actions):
        """
        Args:
            convoluted_contexts: (None, self.hparams.context_dim * num_actions)
        """
        # print(f'%%%%%%%%%%%%. Running NeuralBanditModelV2.net_impure_fn(). %%%%%%%%%%%%.')
        net_structure = []
        for num_units in self.hparams.layer_sizes:
            net_structure.append(
                hk.Linear(
                    num_units, w_init=hk.initializers.UniformScaling(self.hparams.s_init) 
                    )
                ) 
            if self.hparams.layer_n: 
                net_structure.append(hk.LayerNorm(axis=1, create_scale=True, create_offset=True))

            net_structure.append(self.hparams.activation) 

        net_structure.append(
                hk.Linear(1, w_init=hk.initializers.UniformScaling(self.hparams.s_init) )
            )
        
        mlp = hk.Sequential(net_structure) 

        convoluted_contexts = self.action_convolution(contexts, actions)
        return mlp(convoluted_contexts)

    def out_impure_fn(self, params, contexts, actions):
        '''
        type(out_network):Traced<ShapedArray(float32[2,1])>with<DynamicJaxprTrace(level=5/0)>
        out_network:Traced<ShapedArray(float32[2,1])>with<DynamicJaxprTrace(level=5/0)>
        type(out_network):Traced<ShapedArray(float32[1,1])>with<DynamicJaxprTrace(level=4/0)>
        out_network:Traced<ShapedArray(float32[1,1])>with<DynamicJaxprTrace(level=4/0)>
        type(out_network):Traced<ShapedArray(float32[1,1])>with<DynamicJaxprTrace(level=1/0)>
        out_network:Traced<ShapedArray(float32[1,1])>with<DynamicJaxprTrace(level=1/0)>
        '''
        # print(f'%%%%%%%%%%%%. Running NeuralBanditModelV2.out_impure_fn(). %%%%%%%%%%%%.')
        out_network = self.nn.apply(params, contexts, actions)
        # print(f'type(out_network):{out_network}')
        # print(f'out_network:{out_network}')
        return out_network

        # return self.nn.apply(params, contexts, actions)

    '''
    The grad_out_impure_fn function returns gradients of the output with respect to 
    the neural network parameters for given contexts and actions. s
    Specifically, it computes the Jacobian matrix of the network's output function, 
    which represents how changes in the parameters would affect the output predictions for the provided data. 
    The output is structured to match the input batch size and parameter count, 
    formatted as a two-dimensional array where each row corresponds to a sample from contexts, 
    and columns represent the flattened gradients for all parameters.
    '''
    def grad_out_impure_fn(self, params, contexts, actions):
        """
        Args:
            params: Network parameters 
            convoluted_contexts: (None, context_dim * num_actions)

        Return:
            grad_params: (None, p)
        """
        # grad_params = []
        # for gradp in jax.tree_leaves(jax.jacrev(self.out)(params, convoluted_contexts)):
        #     grad_params.append(gradp.reshape(convoluted_contexts.shape[0], -1))  
        # return jnp.hstack(grad_params)
        '''
        for example, acts= [0,2,1], self.hparams.num_actions =3
        after onehot encoding, acts = 
        [
        [1, 0, 0],  # Action 0 encoded
        [0, 0, 1],  # Action 2 encoded
        [0, 1, 0]   # Action 1 encoded
        ]
        When the line acts = jax.nn.one_hot(actions, self.hparams.num_actions)[:,None,:] is executed, 
        it transforms this 2D array into a 3D array by adding a new axis, resulting in:
        [
        [[1, 0, 0]],  # Action 0 encoded
        [[0, 0, 1]],  # Action 2 encoded
        [[0, 1, 0]]   # Action 1 encoded
        ]
        '''
        # print(f'%%%%%%%%%%%%. Running NeuralBanditModelV2.grad_out_impure_fn(). %%%%%%%%%%%%.')

        acts = jax.nn.one_hot(actions, self.hparams.num_actions)[:,None,:]
        '''
        This line creates a tensor ker which is a 3-dimensional identity matrix 
        with dimensions extended by None at the beginning, 
        to match the 3D structure needed for subsequent operations. 
        Specifically, jnp.eye(self.hparams.context_dim) creates a 
        2D identity matrix of size context_dim x context_dim, 
        and [None,:,:] adds a new dimension at the start, making it a 3D tensor. 
        This tensor can be used to select or manipulate specific dimensions 
        in subsequent tensor operations, 
        especially when combined with broadcasting.
        '''
        #ker = lamda*I
        ker = jnp.eye(self.hparams.context_dim)[None,:,:]
        '''
        The line sel = jnp.kron(acts, ker) computes the Kronecker product of acts and ker. 
        Given acts is the one-hot encoded actions expanded into a 3D tensor and ker is an identity matrix expanded 
        into a 3D tensor, jnp.kron combines these tensors in a way that effectively prepares the structure for applying 
        context-specific adjustments based on the selected actions. The result, sel, 
        is a tensor that will be used to filter or modify the gradients based on the specific actions taken.
        '''
        sel = jnp.kron(acts, ker) # (None, context_dim, context_dim * num_actions, :)
        '''
        This line uses jax.jacrev, a function to compute the Jacobian (the vector of partial derivatives) of self.out 
        with respect to params, evaluated at the given contexts and actions. 
        It effectively calculates how small changes in the parameters affect the output of the neural network, 
        essential for understanding the model's sensitivity to its parameters and for gradient-based optimization.
        '''
        # self.out = jax.jit(self.out_impure_fn) 
        # \nabla_{W_{t}}f_{W_{t}}(x_t,a_t)
        grad_params = jax.jacrev(self.out)(params, contexts, actions) 
        # return grad_params

        grads = []
        for key in grad_params:
            if key == 'linear': # extract the weights for the chosen actions only. 
                u = grad_params[key]['w'] 
                # print(f'u:{u}')
                v = jnp.sum(jnp.multiply(u, sel[:,:,:,None]), axis=2) # (None, context_dim, :)
                # print(f'v:{v}')
                grads.append(v.reshape(contexts.shape[0],-1))

                grads.append(grad_params[key]['b'].reshape(contexts.shape[0], -1)) 
            else:
                for p in jax.tree_leaves(grad_params[key]):
                    grads.append(p.reshape(contexts.shape[0], -1))  
                
        return jnp.hstack(grads)

    def action_convolution_impure_fn(self, contexts, actions):
        return action_convolution(contexts, actions, self.hparams.num_actions)

    '''
    NeuralBanditModel's loss_impure_fn
    def loss_impure_fn(self, params, context, action, reward):
        """
        Args:
            context: An array of context, (None, self.hparams.context_dim)
            action: An array of one-hot action vectors, 1 for selected action and 0 other wise, (None, self.hparams.num_actions)
            reward: An array of reward vectors, (None, self.hparams.num_actions)
        """
        # self.out = jax.jit(self.out_impure_fn) 
        preds = self.out(params, context) 

        squared_loss = 0.5 * jnp.mean(jnp.sum(action * jnp.square(preds - reward), axis=1), axis=0)
        reg_loss = 0.5 * self.hparams.lambd * sum(
                jnp.sum(jnp.square(param)) for param in jax.tree_leaves(params) 
            )

        return squared_loss + reg_loss
    '''
 
    
    '''
    Difference between the loss_impure_fn of NeuralBanditModel and NeuralBanditModelV2:
    First Version: Actions are provided as one-hot encoded vectors, and rewards as vectors corresponding to each action. 
    The loss is computed by considering the squared difference between the model's predictions and the rewards, 
    but only for the selected actions (as indicated by the one-hot encoding).

    Second Version: Actions are likely provided as indices (not one-hot encoded), 
    and rewards are for the chosen actions only. The predictions might be directly for the chosen actions, 
    simplifying the calculation of squared loss to a direct comparison between predicted rewards and actual rewards for those actions
    '''

    def loss_impure_fn(self, params, contexts, actions, rewards):
        """
        Args:
            contexts: An array of context, (None, self.hparams.context_dim)
            actions: An array of actions, (None,)
            rewards: An array of rewards for the chosen actions, (None,)
        """
        # print(f'#######Before Running loss_impure_fn in NeuralBanditModelV2...........')
        # print("Predictions shape:", preds.shape)
        # print("Rewards shape:", rewards.shape)

        preds = self.out(params, contexts, actions) 
        # print(f'#######Running loss_impure_fn in NeuralBanditModelV2...........')
        # print("Predictions shape:", preds.shape)
        # print("Rewards shape:", rewards.shape)
        # sys.exit()
        debugging = False
        # debugging = True
        if debugging == False:
            squared_loss = 0.5 * jnp.mean( jnp.square(preds.ravel() - rewards.ravel()) )
            reg_loss = 0.5 * self.hparams.lambd * sum(
                    jnp.sum(jnp.square(param)) for param in jax.tree_leaves(params) 
                )
        else:
        # debugging
            squared_loss = 0.05 
            reg_loss =  0
        return squared_loss + reg_loss 

    def clone(self):
        return copy.deepcopy(self)
    

    def init(self, seed): 
        key = jax.random.PRNGKey(seed)
        key, subkey = jax.random.split(key)

        context = jax.random.normal(key, (1, self.hparams.context_dim))
        # this is what is added compared with NeuralBanditModel
        action = jax.random.randint(subkey, shape=(1,), minval=0, maxval=self.hparams.num_actions) 
        # self.nn.init(): returns the initializd parameters  for the network. 
        # these parameters include weights and biass for each layer, structured in a nested dictionary 
        self.params = self.nn.init(subkey, context, action) 
        # print(f'self.params: {self.params}')
        self.opt_state = self.optimizer.init(self.params)

    
    def update_impure_fn(self, params, opt_state, contexts, actions, rewards ):
        """
        Args:
            contexts: An array of contexts, (None, self.hparams.context_dim)
            actions: An array of actions, (None, )
            rewards: An array of rewards for the chosen actions, (None,)
        """
        # The gradient of the loss function with respect to the parameters (grads = jax.grad(self.loss)) is computed, and the optimizer (like Adam) updates the parameters accordingly.
        # print(f'##### Before Running update_impure_fn in NeuralBanditModelV2...........')
        # print("Predictions shape:", preds.shape)
        # print("Rewards shape:", rewards.shape)
        grads = jax.grad(self.loss)(params, contexts, actions, rewards )
        updates, opt_state = self.optimizer.update(grads, opt_state)
        new_params = optax.apply_updates(params, updates)
        return new_params, opt_state

    # data is a BanditDataset 
    def train(self, data, num_steps):
        print('(((((((Running train() in NeuralBanditModelV2)))))))')
        if self.hparams.verbose:
            print('*******Training {} for {} steps.'.format(self.name, num_steps)) 
            
        params, opt_state = self.params, self.opt_state 
        # flags.DEFINE_integer('num_steps', 100, 'Number of steps to train NN.') 
        # The concept of num_steps in the train() function refers to the number of gradient update steps 
        # or iterations that the model will go through during training. 
        # It controls how many times the model processes batches of data and updates its parameters.

        for step in range(num_steps):
            """
            Return:
                x: (batch_size, context_dim)
                a: (batch_size, )
                y: (batch_size, )
            """
            # flags.DEFINE_bool('data_rand', True, 'Where randomly sample a data batch or  use the latest samples in the buffer' )

            x,a,y = data.get_batch(self.hparams.batch_size, self.hparams.data_rand) #(None,d), (None,), (None,)
            # x,a,y = data.get_batch_with_weights(self.hparams.batch_size)
            # print(f'(x,a,y) from get_batch():')
            # print(f'x:{x}')
            # print(f'a:{a}')
            # print(f'y.shape:{y.shape}')s
            # sys.exit()
            # self.update = jax.jit(self.update_impure_fn) 
            # update_impure_fn(self, params, opt_state, contexts, actions, rewards)  
           
           
            
            params, opt_state = self.update(params, opt_state, x,a,y ) 
            

            # print the step loss
            # flags.DEFINE_integer('freq_summary', 10, 'Summary frequency')
            if step % self.hparams.freq_summary == 0 and self.hparams.verbose:
                # self.loss = jax.jit(self.loss_impure_fn)
                # loss_impure_fn(self, params, contexts, actions, rewards)

                cost = self.loss(params, x,a,y)
                print('{} | step: {} | loss: {}'.format(self.name, step, cost))
            # if step == 0 or step == num_steps-1:
            #     print(f'$$$$$$$$$ Number step-{step} of NeuralBanditModelV2.train() Finished $$$$$$$$$')       
        
        self.params, self.opt_state = params, opt_state
        # print(f'self.params:{self.params}')
        # print(f'self.opt_state:{self.opt_state}')
        # sys.exit()













    '''
    Rather than replacing the existing gradient computation, we can use LOO predictions as an additional signal in the gradient computation. 
 

    1. Use LOO Predictions in Gradient Calculation: 
    If utilize LOO predictions directly, 
    modify the grad_out_impure_fn to take into account the differences between the model’s current predictions and the LOO predictions.

    2. Incorporate LOO Predictions as Regularization: 
    use the LOO predictions as a regularization term in loss function, 
    ensuring that model’s parameters are updated not only to minimize the standard loss but also to stay consistent with the LOO predictions.
    '''
    # def grad_out_impure_cp(self, params, contexts, actions, loo_preds=None):
    #     """
    #     Args:
    #         params: Network parameters 
    #         contexts: (None, context_dim)
    #         actions: (None,)
    #         loo_preds: (None,) - Leave-One-Out predictions, optional
    #     """
    #     acts = jax.nn.one_hot(actions, self.hparams.num_actions)[:, None, :]
    #     ker = jnp.eye(self.hparams.context_dim)[None, :, :]
    #     sel = jnp.kron(acts, ker)  # (None, context_dim, context_dim * num_actions, :)

    #     # Compute gradients with respect to parameters
    #     grad_params = jax.jacrev(self.out)(params, contexts, actions)

    #     grads = []
    #     for key in grad_params:
    #         if key == 'linear':  # extract the weights for the chosen actions only.
    #             u = grad_params[key]['w']
    #             v = jnp.sum(jnp.multiply(u, sel[:, :, :, None]), axis=2)  # (None, context_dim, :)
    #             grads.append(v.reshape(contexts.shape[0], -1))
    #             grads.append(grad_params[key]['b'].reshape(contexts.shape[0], -1))
    #         else:
    #             for p in jax.tree_leaves(grad_params[key]):
    #                 grads.append(p.reshape(contexts.shape[0], -1))

    #     # Combine the gradients
    #     combined_grads = jnp.hstack(grads)

    #     # If LOO predictions are provided, adjust the gradients accordingly
    #     if loo_preds is not None:
    #         # Adjust gradients based on the difference between current predictions and LOO predictions
    #         # For example, we could add or subtract a fraction of the LOO difference to/from the gradient
    #         preds = self.out(params, contexts, actions).ravel()  # Current model predictions
    #         diff = preds - loo_preds
    #         adjustment = jnp.outer(diff, jnp.ones_like(combined_grads[0]))  # Create adjustment based on LOO diffs
    #         combined_grads += adjustment  # Modify the gradients using LOO predictions

    #     return combined_grads