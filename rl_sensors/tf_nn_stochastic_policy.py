import random

import numpy as np
import tensorflow as tf


class TFNeuralNetStochasticPolicyOTPSensor:
    def __init__(self, num_input, init_learning_rate=1e-6, min_learning_rate=1e-10, learning_rate_N_max=10000,
                 sigma=None, shuffle=True, batch_size=1, init_pos=None, non_linearity=tf.nn.tanh, clip_norm=5.0):
        self._sess = tf.Session()
        dtype = tf.float32
        self._states = tf.placeholder(dtype, (None, num_input), name="states")

        self._init_learning_rate = init_learning_rate
        self._min_learning_rate = min_learning_rate
        self._learning_rate_N_max = learning_rate_N_max
        self._learning_rate = tf.placeholder(dtype, shape=[])
        self._clip_norm = clip_norm

        self._mu_theta_hidden = 800
        self._sigma_theta_hidden = 100
        self._layer1_hidden = 1600
        self._layer2_hidden = self._mu_theta_hidden

        with tf.name_scope("network_variables"):
            # policy parameters
            self._mu_theta = tf.get_variable("mu_theta", [2, self._mu_theta_hidden],
                                             initializer=tf.zeros_initializer(), dtype=dtype)

            if sigma is None:
                self._sigma_theta = tf.get_variable("sigma_theta", [2, self._sigma_theta_hidden],
                                                    initializer=tf.zeros_initializer(), dtype=dtype)

            # neural featurizer parameters
            self._W1 = tf.get_variable("W1", [num_input, self._layer1_hidden],
                                       initializer=tf.random_normal_initializer(), dtype=dtype)
            self._b1 = tf.get_variable("b1", [self._layer1_hidden],
                                       initializer=tf.constant_initializer(0), dtype=dtype)
            self._h1 = non_linearity(tf.matmul(self._states, self._W1) + self._b1)
            self._W2 = tf.get_variable("W2", [self._layer1_hidden, self._layer2_hidden],
                                       initializer=tf.random_normal_initializer(stddev=0.1), dtype=dtype)
            self._b2 = tf.get_variable("b2", [self._layer2_hidden],
                                       initializer=tf.constant_initializer(0), dtype=dtype)
            self._phi = non_linearity(tf.matmul(self._h1, self._W2) + self._b2)

        self._mu = tf.matmul(self._phi, tf.transpose(self._mu_theta))

        if sigma is None:
            self._sigma = tf.reduce_sum(self._sigma_theta, 1)
            self._sigma = tf.reshape(tf.exp(self._sigma), [1, 2])
        else:
            self._sigma = tf.constant(sigma, dtype=dtype)

        self._optimizer = tf.train.GradientDescentOptimizer(learning_rate=self._learning_rate)
        # self._optimizer = tf.train.MomentumOptimizer(learning_rate=self._learning_rate, momentum=0.9, use_nesterov=True)
        # self._optimizer = tf.train.AdagradOptimizer(learning_rate=init_learning_rate)
        # self._optimizer = tf.train.AdadeltaOptimizer()
        # self._optimizer = tf.train.RMSPropOptimizer(learning_rate=init_learning_rate, decay=0.9)

        self._discounted_rewards = tf.placeholder(dtype, (None, 1), name="discounted_rewards")
        self._taken_actions = tf.placeholder(dtype, (None, 2), name="taken_actions")

        # self._network_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="network_variables")
        # self.reg_loss = tf.reduce_sum([tf.reduce_sum(tf.square(x)) for x in self._network_variables])

        # we'll get the policy gradient by using -log(pdf), where pdf is the PDF of the Normal distribution
        self._loss = -tf.log(tf.sqrt(1/(2 * np.pi * self._sigma**2)) * tf.exp(-(self._taken_actions - self._mu)**2/(2 * self._sigma**2))) * self._discounted_rewards

        self._gradients, variables = zip(*self._optimizer.compute_gradients(self._loss))
        self._gradients, _ = tf.clip_by_global_norm(self._gradients, self._clip_norm)
        self._train_op = self._optimizer.apply_gradients(zip(self._gradients, variables))
        # self._train_op = self._optimizer.minimize(self._loss)

        self._sess.run(tf.global_variables_initializer())

        self._all_rewards = []
        self._max_reward_length = 1000000

        self._sigma_in = sigma
        self._num_input = num_input
        self._shuffle = shuffle
        self._batch_size = batch_size
        self._init_pos = init_pos
        self.reset_location()

    def reset_location(self):
        if self._init_pos is None:
            self._init_x = 10000 * random.random() - 5000
            self._init_y = 10000 * random.random() - 5000
        else:
            self._init_x = self._init_pos[0]
            self._init_y = self._init_pos[1]
        self._initial_location = [self._init_x, self._init_y]
        self._current_location = self._initial_location
        self._sensor_actions = []
        self._sensor_sigmas = []

    def update_location(self, system_state):
        # Gaussian policy
        mu, sigma = self._sess.run([self._mu, self._sigma], feed_dict={
            self._states: np.reshape(system_state, (1, self._num_input))
        })
        delta = np.random.normal(mu, sigma)

        self._sensor_sigmas.append(sigma)
        self._sensor_actions.append(delta)
        new_x = self._current_location[0] + delta[0][0]
        new_y = self._current_location[1] + delta[0][1]

        new_x = np.clip(new_x, -50000, 50000)
        new_y = np.clip(new_y, -50000, 50000)

        self._current_location = [new_x, new_y]

    def get_current_location(self):
        return self._current_location

    def get_weights(self):
        return []  # TODO for now

    def get_sigmas(self):
        return self._sensor_sigmas

    def _gen_learning_rate(self, iteration, l_max, l_min, N_max):
        if iteration > N_max:
            return l_min
        alpha = 2 * l_max
        beta = np.log((alpha / l_min - 1)) / N_max
        return alpha / (1 + np.exp(beta * iteration))

    def update_parameters(self, iteration, discounted_return, episode_states):
        episode_actions = self._sensor_actions
        learning_rate = self._gen_learning_rate(iteration, l_max=self._init_learning_rate,
                                                l_min=self._min_learning_rate, N_max=self._learning_rate_N_max)

        # reduce gradient variance by normalization
        self._all_rewards += discounted_return.tolist()
        self._all_rewards = self._all_rewards[:self._max_reward_length]
        discounted_return -= np.mean(self._all_rewards)
        discounted_return /= np.std(self._all_rewards)

        N = len(episode_states)

        all_samples = []
        for t in range(N-1):
            state  = np.reshape(np.array(episode_states[t]), self._num_input)
            action = episode_actions[t][0]
            reward = [discounted_return[t]]
            sample = [state, action, reward]
            all_samples.append(sample)
        if self._shuffle:
            np.random.shuffle(all_samples)

        batches = list(self._minibatches(all_samples, batch_size=self._batch_size))

        for b in range(len(batches)):
            # prepare inputs
            batch = batches[b]
            states = [row[0] for row in batch]
            actions = [row[1] for row in batch]
            rewards = [row[2] for row in batch]

            self._sess.run([self._train_op], feed_dict={
                self._states:             states,
                self._taken_actions:      actions,
                self._discounted_rewards: rewards,
                self._learning_rate:      learning_rate
            })
        return True

    def _minibatches(self, samples, batch_size):
        for i in range(0, len(samples), batch_size):
            yield samples[i:i + batch_size]

    def __str__(self):
        return self.__class__.__name__ + "_" + str(self._num_input) + "_" + str(self._init_learning_rate) + "_" + \
               str(self._min_learning_rate) + "_" + str(self._learning_rate_N_max) + "_" + str(self._sigma_in) + "_" + \
               str(self._shuffle) + "_" + str(self._batch_size)
