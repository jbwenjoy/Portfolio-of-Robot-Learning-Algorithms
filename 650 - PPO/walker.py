from dm_control import suite, viewer
import numpy as np
import scipy
import matplotlib.pyplot as plt
import pandas as pd
from copy import copy
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.optim import Adam

import os


### Important note on references:
# The code is adapted from the following sources:
# - OpenAI Spinning Up Documentation: https://spinningup.openai.com/en/latest/index.html
# - OpenAI Spinning Up PyTorch PPO repo: https://github.com/openai/spinningup/tree/master/spinup/algos/pytorch/ppo
# - ppo_dm_control repo: https://github.com/JhonPool4/ppo_dm_control/blob/master/ppo_utils/policies.py


def multilayer_perceptron(sizes, activation, output_activation=nn.Identity):
    """
    MLP

    :param sizes: sizes of the layers
    :param activation: activation functions
    :param output_activation: activation function for the output layer
    :return: a neural network model
    """
    layers = []
    for j in range(len(sizes)-1):
        act = activation[j] if j < len(sizes)-2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j+1]), act()]
    return nn.Sequential(*layers)


def discount_cumsum(x, discount):
    """
    Magic from rllab for computing discounted cumulative sums of vectors.

    input:
        vector x,
        [x0,
         x1,
         x2]
    output:
        [x0 + discount * x1 + discount^2 * x2,
         x1 + discount * x2,
         x2]
    """
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]


class ActorNN(nn.Module):
    """
    A Gaussian policy implemented using a Multilayer Perceptron (MLP).
    Adapted from OpenAI's implementations.
    """
    def __init__(self, obs_dim, act_dim, hidden_sizes, activation):
        """
        Initialize the MLP Gaussian Actor.

        Parameters:
        - obs_dim (int): The dimension of the observation space.
        - act_dim (int): The dimension of the action space.
        - hidden_sizes (list): List of sizes for each hidden layer.
        - activation (callable): Activation function for the hidden layers.
        """
        super().__init__()

        # Initialize the log standard deviation as a trainable parameter
        # Start with a slightly negative value to ensure a small positive variance
        log_std = -0.5 * np.ones(act_dim, dtype=np.float32)
        self.log_std = torch.nn.Parameter(torch.as_tensor(log_std))

        # Create the MLP network to output the mean values of the Gaussian distribution
        self.mu_net = multilayer_perceptron([obs_dim] + list(hidden_sizes) + [act_dim], activation)

    def gaussian_distri(self, obs):
        """
        Create a Gaussian distribution based on the network output (mean and std).
        """
        mu = self.mu_net(obs)
        std = torch.exp(self.log_std)
        return torch.distributions.normal.Normal(mu, std)

    def action_log_prob_from_distri(self, pi, act):
        """
        Calculate the log probability of an action given a distribution.
        """
        # Sum across the last axis to match the behavior of Torch's Normal distribution
        return pi.log_prob(act).sum(axis=-1)  # Last axis sum needed for Torch Normal distribution

    def forward(self, obs, action=None):
        """
        Ref: https://github.com/JhonPool4/ppo_dm_control/blob/master/ppo_utils/policies.py
        Forward pass through the network, optionally calculating the log probability of an action.

        Parameters:
        - obs: The input observation data.
        - action: The action to compute the log probability for.

        Returns:
        - dist: The Gaussian distribution generated from the observation.
        - logp_a: The log probability of the action (if given), otherwise None.
        """
        dist = self.gaussian_distri(obs)
        logp_a = None

        # action can be None or a tensor
        if action is not None:
            logp_a = self.action_log_prob_from_distri(dist, action)
        return dist, logp_a


class CriticNN(nn.Module):
    """
    The MLP Critic network.
    """

    def __init__(self, obs_dim, hidden_sizes, activation):
        """
        Parameters:
        - obs_dim (int): The dimension of the observation space.
        - hidden_sizes (list): List of sizes for each hidden layer.
        - activation (callable): Activation function for the hidden layers.
        """
        super().__init__()

        # Create an MLP network that outputs a single value representing the estimated value of a state
        # This value network (v_net) will estimate state value for the critic
        self.v_net = multilayer_perceptron([obs_dim] + list(hidden_sizes) + [1], activation)

    def forward(self, obs):
        """
        Forward pass to compute the value of a given observation.
        """
        return torch.squeeze(self.v_net(obs), -1)


class ActorCriticNN(nn.Module):
    """
    A combined Actor-Critic model.
    """
    def __init__(self, obs_dim, act_dim, hidden_sizes, activation=nn.Tanh):
        """
        Parameters:
        - obs_dim (int): The dimension of the observation space.
        - act_dim (int): The dimension of the action space.
        - hidden_sizes (list): List of sizes for each hidden layer.
        - activation (callable): Activation function for the hidden layers.
        """
        super().__init__()

        # Create the policy network (Actor), which generates actions based on observations
        # This will be a Gaussian actor that learns a distribution over actions
        self.pi = ActorNN(obs_dim, act_dim, hidden_sizes, activation)

        # Create the value function
        self.vf = CriticNN(obs_dim, hidden_sizes, activation)

    def step(self, obs):
        """
        Forward step to sample an action, compute its log probability, and estimate the value of the state.

        Returns:
        - action (numpy array): The sampled action.
        - vf (numpy array): The estimated value of the state.
        - logp_a (numpy array): The log probability of the sampled action.
        """
        with torch.no_grad():  # No need to store gradients for this step
            # Generate the Gaussian distribution for the current observation
            distribution = self.pi.gaussian_distri(obs)
            
            # Sample an action
            action = distribution.sample()

            # Calculate the log probability of the action
            logp_a = self.pi.action_log_prob_from_distri(distribution, action)

            # Estimate the value of the state observation
            vf = self.vf(obs)

        return action.numpy(), vf.numpy(), logp_a.numpy()

    def act(self, obs):
        """
        Return only the action sampled from the policy network.
        """
        return self.step(obs)[0]


class PPOBuffer:
    """
    Buffer for storing trajectory data collected by a PPO agent's interaction with the env.
    Uses Generalized Advantage Estimation (GAE-Lambda) to compute the advantages of state-action pairs.
    """

    def __init__(self, obs_dim, act_dim, size, gamma=0.99, lam=0.95):
        """
        Parameters:
        - obs_dim (int): The dimension of the observation space.
        - act_dim (int): The dimension of the action space.
        - size (int): The maximum number of timesteps to store.
        - gamma (float): Discount factor for rewards.
        - lam (float): Lambda parameter for GAE-Lambda.
        """
        self.obs_buf = np.zeros((size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros((size, act_dim), dtype=np.float32)
        self.adv_buf = np.zeros(size, dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.ret_buf = np.zeros(size, dtype=np.float32)
        self.val_buf = np.zeros(size, dtype=np.float32)
        self.logp_buf = np.zeros(size, dtype=np.float32)
        self.gamma = gamma
        self.lam = lam
        self.ptr = 0
        self.path_start_idx = 0
        self.max_size = size
        self.mean_rews = []
        self.cross_corr_rewards = []

    def store(self, obs, act, rew, val, logp):
        """
        Append one timestep of agent-environment interaction to the buffer.

        Parameters:
        - obs: Observation at the current timestep.
        - act: Action taken by the agent.
        - rew: Reward received after taking the action.
        - val: Value estimate of the current state.
        - logp: Log probability of the action under the policy.
        """
        assert self.ptr < self.max_size  # buffer has to have room so you can store
        self.obs_buf[self.ptr] = obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.val_buf[self.ptr] = val
        self.logp_buf[self.ptr] = logp
        self.ptr += 1

    def compute_cross_correlation_reward(self, path_slice):
        """
        After obtaining the default accumulative reward,
        we also calculate the cross-correlation reward of the angle signals,
        aiming to make the walker walk symmetrically.

        This function will be called when the episode ends (in finish_path() function).
        It should read the leg angles data from the entire episode.
        """
        # Get the left and right observation data from the buffer
        left_angles = self.obs_buf[path_slice, 0:3]  # Note sure if it's 0, 1, 2
        right_angles = self.obs_buf[path_slice, 3:6]  # Also not sure if it's 3, 4, 5

        max_correlations = []
        norm_factors = []
        for i in range(3):
            # Compute cross-correlation at all lags and find the maximum value
            correlation = np.correlate(left_angles[:, i], right_angles[:, i], mode='full')
            max_correlation = np.max(correlation)
            max_correlations.append(max_correlation)

            # Calculate normalization factor from the auto-correlation of both signals
            norm_factor = np.sqrt(np.max(np.correlate(left_angles[:, i], left_angles[:, i], 'full')) *
                                  np.max(np.correlate(right_angles[:, i], right_angles[:, i], 'full')))
            norm_factors.append(norm_factor)

        # Calculate normalized cross-correlation and return the average
        normalized_correlations = np.array(max_correlations) / np.array(norm_factors)
        return np.mean(normalized_correlations)

    def finish_path(self, last_val=0):
        """
        Finalize trajectory data at the end of an epoch or when the trajectory ends prematurely.

        This function computes advantages and rewards-to-go using rewards and value estimates,
        following the Generalized Advantage Estimation (GAE-Lambda) method.

        The parameter `last_value` should be set to 0 if the trajectory ended due to a terminal state.
        Otherwise, it is the estimated value of the final state (V(s_T)), which is used to bootstrap
        the reward-to-go computation, accounting for timesteps beyond the arbitrary episode horizon.

        Args:
            last_value (float): The estimated value for the last state or 0 if terminal.
        """
        # Identify the relevant slice of data within the buffers
        path_slice = slice(self.path_start_idx, self.ptr)

        # Add the final reward and value to the trajectory buffers
        rews = np.concatenate((self.rew_buf[path_slice], [last_val]))
        vals = np.concatenate((self.val_buf[path_slice], [last_val]))

        # Update the average rewards for tracking
        self.mean_rews.append(np.mean(rews))

        # Calculate deltas (TD residuals) and the GAE-Lambda advantage
        deltas = rews[:-1] + self.gamma * vals[1:] - vals[:-1]
        self.adv_buf[path_slice] = discount_cumsum(deltas, self.gamma * self.lam)

        # Compute rewards-to-go, removing the last value to align with the trajectory length
        self.ret_buf[path_slice] = discount_cumsum(rews, self.gamma)[:-1]

        # Compute and add the cross-correlation reward
        cross_corr_reward = self.compute_cross_correlation_reward(path_slice)
        self.cross_corr_rewards.append(cross_corr_reward)

        # Reset path index for the next trajectory
        self.path_start_idx = self.ptr

    def get(self):
        """
        Retrieve all stored data after an epoch.
        Advantages are normalized.(shifted to have mean zero and std one)
        Reset pointers in the buffer.

        Returns:
        - Dictionary containing the data with normalized advantages.
        """
        # Ensure the buffer is full.
        if self.ptr != self.max_size:
            raise ValueError("Buffer is not full")
        
        # Normalize the advantages to have zero mean and unit standard deviation
        adv_mean, adv_std = self.adv_buf.mean(), self.adv_buf.std()
        self.adv_buf = (self.adv_buf - adv_mean) / adv_std

        # Collect all data into a dictionary and convert to Torch tensors
        data = dict(obs=self.obs_buf, act=self.act_buf, ret=self.ret_buf,
                    adv=self.adv_buf, logp=self.logp_buf, mean_rews=self.mean_rews,
                    cross_corr_rewards=self.cross_corr_rewards)

        # Reset mean and cross-correlation rewards
        self.mean_rews = []
        self.cross_corr_rewards = []

        # Reset pointers
        self.ptr, self.path_start_idx = 0, 0

        return {k: torch.as_tensor(v, dtype=torch.float32) for k, v in data.items()}


class PPO:
    def __init__(self, env):
        """
        Initialize the PPO training environment.

        Parameters:
        - env: The environment object the agent will interact with.
        - **hyperparameters: Additional configuration values.
        """
        # Update hyperparameters
        self.init_hyperparams()

        # Initialize Retrieve dimensions from the environment
        self.env = env
        self.obs_dim = self.env.obs_dim
        self.act_dim = self.env.act_dim

        # Initialize the actor-critic model and optimizers
        self.ac_model = ActorCriticNN(self.obs_dim, self.act_dim, self.hidden, self.activation)
        self.policyNN_optimizer = Adam(self.ac_model.pi.parameters(), self.pi_lr)
        self.valfunNN_optimizer = Adam(self.ac_model.vf.parameters(), self.vf_lr)

        # Create a buffer to store training data
        self.buffer = PPOBuffer(self.obs_dim, self.act_dim, self.steps_per_epoch, self.gamma, self.lam)

        # Initialize logging and data storage
        self.logger = {'mean_rew': 0, 'std_rew': 0}
        if not os.path.exists(self.training_path):
            os.makedirs(self.training_path)
            print(f"new directory created: {self.training_path}")

        # Set up the training data file
        self.column_names = ['mean', 'std']
        self.df = pd.DataFrame(columns=self.column_names, dtype=object)

    def compute_loss_pi(self, data):
        """
        Calculate the policy loss using clipped surrogate objective.

        Parameters:
        - data: Training data containing observations, actions, advantages, and log probabilities.

        Returns:
        - loss_pi: Calculated policy loss.
        - pi_info: Additional policy information, including KL divergence, entropy, and clipping fraction.
        """
        # Get specific training data
        obs, act, adv, logp_old = data['obs'], data['act'], data['adv'], data['logp']

        # Evaluate new policy
        act_dist, logp = self.ac_model.pi(obs, act)

        # Calculate the ratio of new to old probabilities and apply clipping
        ratio = torch.exp(logp - logp_old)
        clip_adv = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * adv
        
        # Calculate entropy for exploration
        entropy = act_dist.entropy().mean().item()

        # Policy loss with entropy bonus
        loss_pi = -(torch.min(ratio * adv, clip_adv) + self.coef_ent * entropy).mean()

        # About KL divergence and clipping fraction
        approx_kl = (logp_old - logp).mean().item()
        clipped = ratio.gt(1 + self.clip_ratio) | ratio.lt(1 - self.clip_ratio)
        clipfrac = torch.as_tensor(clipped, dtype=torch.float32).mean().item()
        pi_info = dict(kl=approx_kl, ent=entropy, cf=clipfrac)

        return loss_pi, pi_info

    def compute_loss_vf(self, data):
        """
        Calculate the value function loss.

        Parameters:
        - data: Training data containing observations and returns.

        Returns:
        - loss_vf: Mean squared error between predicted and actual returns.
        """
        # Get specific training data
        obs, ret = data['obs'], data['ret']
        
        # Value function loss
        return ((self.ac_model.vf(obs) - ret) ** 2).mean()

    def update(self):
        """
        Update policy and value networks using collected data.
        """
        # Retrieve and normalize training data from the buffer
        data = self.buffer.get()

        # Write logger reward information
        self.logger['mean_rew'] = data['mean_rews'].mean().item()
        self.logger['std_rew'] = data['mean_rews'].std().item()
        self.logger['mean_cross_corr_rew'] = data['cross_corr_rewards'].mean().item()

        # Train policy NN
        for _ in range(self.train_pi_iters):
            self.policyNN_optimizer.zero_grad()
            loss_pi, pi_info = self.compute_loss_pi(data)
            kl = pi_info['kl']
            if kl > self.target_kl:
                # print(f"Early stop at step {i} due to max kl")
                break
            loss_pi.backward()  # compute grads
            self.policyNN_optimizer.step()  # update parameters

        # Train value function NN
        for _ in range(self.train_vf_iters):
            self.valfunNN_optimizer.zero_grad()
            loss_vf = self.compute_loss_vf(data)
            loss_vf.backward()  # compute grads
            self.valfunNN_optimizer.step()  # update parameters

    def rollout(self):
        """
        Generate a trajectory by running the agent in the environment.

        Returns:
        - epoch_reward: The total reward collected during the epoch.
        """
        # Reset environment parameters
        obs, episode_return, episode_len = self.env.reset(), 0, 0
        epoch_reward = 0

        # Generate training data
        for t in range(self.steps_per_epoch):
            # get action, value function and logprob
            action, value, log_prob = self.ac_model.step(torch.as_tensor(obs, dtype=torch.float32))

            next_obs, reward, done_flag, _ = self.env.step(action)
            episode_return += reward
            episode_len += 1
            epoch_reward += reward

            # Save and log data
            self.buffer.store(obs, action, reward, value, log_prob)

            # Update observation
            obs = copy(next_obs)

            timeout = episode_len == self.max_ep_len
            terminal = done_flag or timeout
            epoch_ended = t == self.steps_per_epoch - 1

            if terminal or epoch_ended:
                if epoch_ended and not terminal:
                    print('Warning: trajectory cut off by epoch at %d steps.' % episode_len, flush=True)
                
                # If not terminal, use bootstrapped value estimate for final state
                if timeout or epoch_ended:
                    _, value, _ = self.ac_model.step(torch.as_tensor(obs, dtype=torch.float32))
                else:
                    value = 0
                self.buffer.finish_path(value)

                # Reset environment parameters
                obs, episode_return, episode_len = self.env.reset(), 0, 0

        # Return the total reward for the epoch, remember to add the cross-correlation reward
        epoch_reward += self.cross_corr_coef * self.buffer.cross_corr_rewards[-1]

        return epoch_reward

    def learn(self):
        """
        Train the PPO agent over multiple epochs.
        """
        # Define lists to store results at every iteration
        mean_r = []

        for epoch in tqdm(range(self.epochs)):
            # Collect traj data and get total reward for one epoch
            epoch_reward = self.rollout()

            # Update policy and value function
            self.update()

            # Append results
            # mean_r.append(self.logger['mean_rew'])
            mean_r.append(epoch_reward)

            # if (epoch + 1) % 10 == 0:
            #     print("\n")
            #     print(f"epochs: {epoch + 1}")
            #     print(f"mean_rew: {self.logger['mean_rew']}")
            #     # print(f"std_ret: {self.logger['std_rew']}")
            #     # print("\n")

            # Plot result and save model every a few steps
            if (epoch + 1) % self.save_freq == 0:
                # Plot reward
                plt.figure()
                plt.plot(mean_r)
                plt.title("Mean Reward")
                plt.xlabel("Epoch")
                plt.ylabel("Mean Reward")
                plt.savefig(os.path.join(self.training_path, "mean_rewards.png"))
                plt.show()

                # # Plot standard deviation reward
                # plt.figure()
                # plt.plot(stdev_r)
                # plt.title("Standard Deviation")
                # plt.xlabel("Epoch")
                # plt.ylabel("Standard Deviation")
                # plt.savefig(os.path.join(self.training_path, "std_deviation.png"))
                # plt.show()

                torch.save(self.ac_model.state_dict(), os.path.join(self.training_path, self.model_filename))
                print("saving model")

            # Reset logger
            self.logger = {'rew_mean': 0, 'rew_std': 0}

            # Adjust lr if necessary
            if (epoch + 1) % self.lr_decay_freq == 0:
                self.pi_lr *= self.lr_gamma
                self.vf_lr *= self.lr_gamma
                self.adjust_lr(self.policyNN_optimizer, self.pi_lr)
                self.adjust_lr(self.valfunNN_optimizer, self.vf_lr)
                print("New pi_lr = %f" % self.pi_lr)
                print("New vf_lr = %f" % self.vf_lr)

    def adjust_lr(self, optimizer, new_lr):
        """
        Adjust the learning rate of an optimizer.
        """
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lr

    def init_hyperparams(self):
        self.epochs = 8000
        self.steps_per_epoch = 1000
        self.max_ep_len = 1000  # 500
        self.gamma = 0.95
        self.lam = 0.95
        self.clip_ratio = 0.06
        self.target_kl = 0.01 * 2.5
        self.coef_ent = 0.001
        self.cross_corr_coef = 10  # weight of the normalized cross-correlation reward, should be large

        self.train_pi_iters = 50
        self.train_vf_iters = 50
        self.pi_lr = 3e-4 * 0.1
        self.vf_lr = 1e-3 * 0.1
        self.lr_gamma = 0.5
        self.lr_decay_freq = 2000

        self.hidden = (128, 128)
        self.activation = [nn.Tanh, nn.ReLU]

        self.flag_render = False

        self.save_freq = 100

        self.training_path = './training/walker'
        self.data_filename = 'data'
        self.model_filename = 'ppo_walker_model_cross_corr.pth'



class WalkerEnv:
    def __init__(self, env):
        """
        Initialize the Walker environment wrapper.

        Parameters:
        - env: The environment to wrap, providing action and observation specifications.
        """
        self.env = env
        self.act_space = env.action_spec()
        self.obs_space = env.observation_spec()
        self.act_dim = self.act_space.shape[0]
        # ori_shape = (self.obs_space['orientations']).shape
        # hei_shape = (self.obs_space['height']).shape
        # vel_shape = (self.obs_space['velocity']).shape
        self.obs_dim = 14 + 1 + 9

        self.state = self.env.reset()
        self.done = False
        self.step_count = 0

    def get_obs_array(self, obs):
        """
        Convert the observation dictionary into a flattened numpy array.

        Parameters:
        - obs (dict): Dictionary containing different aspects of the observation.

        Returns:
        - np.ndarray: Flattened array with all observation data concatenated.
        """
        parsed = np.concatenate([np.ravel(obs[key]) for key in obs])
        return parsed

    def reset(self):
        """
        Reset the environment to start a new episode.
        """
        self.state = self.env.reset()
        self.step_count = 0
        return self.get_obs_array(self.state.observation)

    def get_reward(self, parsed_obs):
        """
        Compute the reward for the current step based on parsed observations.
        Unparsed observation has 14 elements in orientations, 1 element in height and 9 elements in velocity
        Default reward
        """
        reward = self.state.reward
        # reward += parsed_obs[14] * 0.05
        return reward

    def step(self, action):
        """
        Perform a step in the environment using the provided action.

        Parameters:
        - action: Action to take in the environment.

        Returns:
        - obs (np.ndarray): Observation after taking the action.
        - reward (float): Reward obtained for the current step.
        - done (bool): Whether the episode has ended.
        - info (dict): Additional information (currently empty).
        """
        self.step_count += 1
        self.state = self.env.step(action)
        obs = self.get_obs_array(self.state.observation)
        reward = self.get_reward(obs)
        done = self.state.last()
        # print("\robs: ", obs)
        return obs, reward, done, {}


if __name__ == '__main__':

    # Setup random numbers
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    np.random.seed(42)
    torch.backends.cudnn.deterministic = True

    # Setup walker environment
    r0 = np.random.RandomState(42)
    # env = suite.load('walker', 'walk', task_kwargs={'random': r0})
    env = suite.load('walker', 'walk', task_kwargs={'random': r0})
    walker_env = WalkerEnv(env)

    # Retrieve num of dims of observation and input
    U = env.action_spec()
    udim = U.shape[0]
    X = env.observation_spec()
    xdim = 14 + 1 + 9

    # Have a look at the joints
    obs_spec = env.observation_spec()

    # Visualize a random controller
    RANDOM_CTRL_VISUALIZATION_FLAG = False
    if RANDOM_CTRL_VISUALIZATION_FLAG:
        def u(dt):
            # return np.random.uniform(low=U.minimum, high=U.maximum, size=U.shape)
            test = np.random.uniform(low=-500, high=500)
            u_ = np.array([0, 0, 0, test, 0, 0])  # [r_hip, r_knee, r_ankle, l_hip, l_knee, l_ankle]
            return u_
        # u = np.zeros(udim)
        # u[0] = 1
        # x = np.zeros(xdim)
        viewer.launch(env, policy=u)

    # Train
    agent = PPO(walker_env)
    TRAIN_FLAG = False
    RESUME_TRAINING_FLAG = False
    if TRAIN_FLAG:
        if RESUME_TRAINING_FLAG:
            agent.ac_model.load_state_dict(torch.load('./training/walker/ppo_walker_model_cross_corr.pth'))
        agent.learn()

    #####

    VISUALIZE_FLAG = True
    if VISUALIZE_FLAG:
        # Load trained model
        agent.ac_model.load_state_dict(torch.load('./training/walker/ppo_walker_model.pth'))
        agent.ac_model.eval()

        # Define the action selection function for the viewer
        def choose_action(time_step):
            """
            Determines the next action to take based on the given time step.

            If it's the first time step, an array of zeros matching the action shape is returned.
            Otherwise, it utilizes the agent's policy model to select an action given the current observation.

            Args:
                time_step: A TimeStep object containing observation data and information about whether it's the first step.

            Returns:
                np.ndarray: An array representing the selected action.
            """
            if time_step.first():
                # Initial action is an array of zeros matching the shape of the environment's action space
                action = np.zeros(env.action_spec().shape)
            else:
                # Extract the observation from the time step
                observation = walker_env.get_obs_array(time_step.observation)

                # Convert the observation to a tensor for model processing
                observation_tensor = torch.tensor(observation, dtype=torch.float32)

                # Predict the action using the policy model
                with torch.no_grad():
                    action_tensor, _, _ = agent.ac_model.step(observation_tensor)

                # Ensure the action is in NumPy array format
                action = action_tensor.numpy() if isinstance(action_tensor, torch.Tensor) else action_tensor

            return action

        # Launch the viewer
        viewer.launch(env, choose_action)
