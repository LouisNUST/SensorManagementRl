import tensorflow as tf
import numpy as np
import sklearn
import random

import sklearn.pipeline

from sklearn.kernel_approximation import RBFSampler


class TFRandomFeaturesStochasticPolicyAgent:
    def __init__(self, env, num_input=100, init_learning_rate=0.0001):
        # Feature Preprocessing: Normalize to zero mean and unit variance
        # We use a few samples from the observation space to do this
        self._env = env
        observation_examples = np.array([env.observation_space.sample() for x in range(10000)])
        self._scaler = sklearn.preprocessing.StandardScaler()
        self._scaler.fit(observation_examples)

        # Used to converte a state to a featurizes represenation.
        # We use RBF kernels with different variances to cover different parts of the space
        self._featurizer = sklearn.pipeline.FeatureUnion([
            ("rbf1", RBFSampler(gamma=5.0, n_components=25)),
            ("rbf2", RBFSampler(gamma=2.0, n_components=25)),
            ("rbf3", RBFSampler(gamma=1.0, n_components=25)),
            ("rbf4", RBFSampler(gamma=0.5, n_components=25))
        ])
        self._featurizer.fit(self._scaler.transform(observation_examples))

        self._sess = tf.Session()
        self._states = tf.placeholder(tf.float32, (None, num_input), name="states")

        self._init_learning_rate = init_learning_rate
        self._learning_rate = tf.placeholder(tf.float32, shape=[])

        # policy parameters
        self._mu_theta = tf.get_variable("mu_theta", [1, 100],
                                         initializer=tf.zeros_initializer())
        self._sigma_theta = tf.get_variable("sigma_theta", [1, 100],
                                            initializer=tf.zeros_initializer())

        self._mu = tf.matmul(self._states, tf.transpose(self._mu_theta))
        self._sigma = tf.matmul(self._states, tf.transpose(self._sigma_theta))
        self._sigma = tf.nn.softplus(self._sigma) + 1e-5

        self._optimizer = tf.train.GradientDescentOptimizer(learning_rate=self._learning_rate)

        self._discounted_rewards = tf.placeholder(tf.float32, (None,), name="discounted_rewards")
        self._taken_actions = tf.placeholder(tf.float32, (None, 1), name="taken_actions")

        # we'll get the policy gradient by using -log(pdf), where pdf is the PDF of the Normal distribution
        self._loss = -tf.log(tf.sqrt(1/(2 * np.pi * self._sigma**2)) * tf.exp(-(self._taken_actions - self._mu)**2/(2 * self._sigma**2)))

        self._gradients = self._optimizer.compute_gradients(self._loss)
        for i, (grad, var) in enumerate(self._gradients):
            if grad is not None:
                self._gradients[i] = (grad * self._discounted_rewards, var)
        self._train_op = self._optimizer.apply_gradients(self._gradients)

        self._sess.run(tf.global_variables_initializer())

        self._num_input = num_input
        # rollout buffer
        self._state_buffer  = []
        self._reward_buffer = []
        self._action_buffer = []
        # record reward history for normalization
        self._all_rewards = []
        self._max_reward_length = 1000000
        self._discount_factor = 0.99

    def _featurize_state(self, state):
        scaled = self._scaler.transform(np.array(state).reshape(1, len(state)))
        featurized = self._featurizer.transform(scaled)
        return featurized[0]

    def sample_action(self, system_state):
        # Gaussian policy
        system_state = self._featurize_state(system_state)
        mu, sigma = self._sess.run([self._mu, self._sigma], feed_dict={
            self._states: np.reshape(system_state, (1, self._num_input))
        })
        action = np.random.normal(mu, sigma)
        action = np.clip(action, self._env.action_space.low[0], self._env.action_space.high[0])
        return action

    def store_rollout(self, state, action, reward):
        self._action_buffer.append(action)
        self._reward_buffer.append(reward)
        state = self._featurize_state(state)
        self._state_buffer.append(state)

    def update_model(self, iteration):
        N = len(self._reward_buffer)
        r = 0 # use discounted reward to approximate Q value

        # compute discounted future rewards
        discounted_rewards = np.zeros(N)
        for t in reversed(range(N)):
            # future discounted reward from now on
            r = self._reward_buffer[t] + self._discount_factor * r
            discounted_rewards[t] = r

        # reduce gradient variance by normalization
        self._all_rewards += discounted_rewards.tolist()
        self._all_rewards = self._all_rewards[:self._max_reward_length]
        discounted_rewards -= np.mean(self._all_rewards)
        discounted_rewards /= np.std(self._all_rewards)

        learning_rate = self._gen_learning_rate(iteration, l_max=self._init_learning_rate, l_min=1E-9, N_max=2000)

        for t in range(N-1):

            # prepare inputs
            states  = np.reshape(np.array(self._state_buffer[t]), (1, self._num_input))
            action = np.array(self._action_buffer[t])
            rewards = np.array([discounted_rewards[t]])

            # perform one update of training
            self._sess.run([self._train_op], {
                self._states:             states,
                self._taken_actions:      action,
                self._discounted_rewards: rewards,
                self._learning_rate:      learning_rate
            })
        self._clean_up()

    def _gen_learning_rate(self, iteration, l_max, l_min, N_max):
        if iteration > N_max:
            return l_min
        alpha = 2 * l_max
        beta = np.log((alpha / l_min - 1)) / N_max
        return alpha / (1 + np.exp(beta * iteration))

    def _clean_up(self):
        self._state_buffer  = []
        self._reward_buffer = []
        self._action_buffer = []


class TFNeuralNetStochasticPolicyAgent:
    def __init__(self, env, num_input, init_learning_rate=1e-6):
        self._env = env
        self._sess = tf.Session()
        self._states = tf.placeholder(tf.float32, (None, num_input), name="states")

        self._init_learning_rate = init_learning_rate
        self._learning_rate = tf.placeholder(tf.float32, shape=[])

        # policy parameters
        self._mu_theta = tf.get_variable("mu_theta", [1, 100],
                                              initializer=tf.zeros_initializer())
        self._sigma_theta = tf.get_variable("sigma_theta", [1, 100],
                                            initializer=tf.zeros_initializer())

        # neural featurizer parameters
        self._W1 = tf.get_variable("W1", [num_input, 500],
                                   initializer=tf.random_normal_initializer())
        self._b1 = tf.get_variable("b1", [500],
                                   initializer=tf.constant_initializer(0))
        self._h1 = tf.nn.tanh(tf.matmul(self._states, self._W1) + self._b1)
        self._W2 = tf.get_variable("W2", [500, 1000],
                                   initializer=tf.random_normal_initializer(stddev=0.1))
        self._b2 = tf.get_variable("b2", [1000],
                                   initializer=tf.constant_initializer(0))
        # self._phi = tf.matmul(self._h1, self._W2) + self._b2
        self._h2 = tf.nn.tanh(tf.matmul(self._h1, self._W2) + self._b2)

        self._W3 = tf.get_variable("W3", [1000, 100],
                                   initializer=tf.random_normal_initializer(stddev=0.1))
        self._b3 = tf.get_variable("b3", [100],
                                   initializer=tf.constant_initializer(0))
        self._phi = tf.matmul(self._h2, self._W3) + self._b3

        self._mu = tf.matmul(self._phi, tf.transpose(self._mu_theta))
        self._sigma = tf.matmul(self._phi, tf.transpose(self._sigma_theta))
        self._sigma = tf.nn.softplus(self._sigma) + 1e-5

        self._optimizer = tf.train.GradientDescentOptimizer(learning_rate=self._learning_rate)

        self._discounted_rewards = tf.placeholder(tf.float32, (None,), name="discounted_rewards")
        self._taken_actions = tf.placeholder(tf.float32, (None, 1), name="taken_actions")

        # we'll get the policy gradient by using -log(pdf), where pdf is the PDF of the Normal distribution
        self._loss = -tf.log(tf.sqrt(1/(2 * np.pi * self._sigma**2)) * tf.exp(-(self._taken_actions - self._mu)**2/(2 * self._sigma**2)))

        self._gradients = self._optimizer.compute_gradients(self._loss)
        for i, (grad, var) in enumerate(self._gradients):
            if grad is not None:
                self._gradients[i] = (grad * self._discounted_rewards, var)
        self._train_op = self._optimizer.apply_gradients(self._gradients)

        self._sess.run(tf.global_variables_initializer())

        self._num_input = num_input
        # rollout buffer
        self._state_buffer  = []
        self._reward_buffer = []
        self._action_buffer = []
        # record reward history for normalization
        self._all_rewards = []
        self._max_reward_length = 1000000
        self._discount_factor = 0.99

    def sample_action(self, system_state):
        # Gaussian policy
        mu, sigma = self._sess.run([self._mu, self._sigma], feed_dict={
            self._states: np.reshape(system_state, (1, self._num_input))
        })
        action = np.random.normal(mu, sigma)
        action = np.clip(action, self._env.action_space.low[0], self._env.action_space.high[0])
        return action

    def store_rollout(self, state, action, reward):
        self._action_buffer.append(action)
        self._reward_buffer.append(reward)
        self._state_buffer.append(state)

    def update_model(self, iteration):
        N = len(self._reward_buffer)
        r = 0 # use discounted reward to approximate Q value

        # compute discounted future rewards
        discounted_rewards = np.zeros(N)
        for t in reversed(range(N)):
            # future discounted reward from now on
            r = self._reward_buffer[t] + self._discount_factor * r
            discounted_rewards[t] = r

        # reduce gradient variance by normalization
        self._all_rewards += discounted_rewards.tolist()
        self._all_rewards = self._all_rewards[:self._max_reward_length]
        discounted_rewards -= np.mean(self._all_rewards)
        discounted_rewards /= np.std(self._all_rewards)

        learning_rate = self._gen_learning_rate(iteration, l_max=self._init_learning_rate, l_min=1E-10, N_max=2000)

        for t in range(N-1):

            # prepare inputs
            states  = np.reshape(np.array(self._state_buffer[t]), (1, self._num_input))
            action = np.array(self._action_buffer[t])
            rewards = np.array([discounted_rewards[t]])

            # perform one update of training
            self._sess.run([self._train_op], {
                self._states:             states,
                self._taken_actions:      action,
                self._discounted_rewards: rewards,
                self._learning_rate:      learning_rate
            })
        self._clean_up()

    def _gen_learning_rate(self, iteration, l_max, l_min, N_max):
        if iteration > N_max:
            return l_min
        alpha = 2 * l_max
        beta = np.log((alpha / l_min - 1)) / N_max
        return alpha / (1 + np.exp(beta * iteration))

    def _clean_up(self):
        self._state_buffer  = []
        self._reward_buffer = []
        self._action_buffer = []
