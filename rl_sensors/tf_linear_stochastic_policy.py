import random

import numpy as np
import tensorflow as tf


class TFLinearStochasticPolicyOTPSensor:
    def __init__(self, num_input, init_learning_rate=1e-6, min_learning_rate=1e-10, learning_rate_N_max=10000,
                 init_sigma=1., init_pos=None, optimizer=tf.train.GradientDescentOptimizer):
        self._sess = tf.Session()
        dtype = tf.float32
        self._states = tf.placeholder(dtype, [1, num_input], name="states")

        self._init_learning_rate = init_learning_rate
        self._min_learning_rate = min_learning_rate
        self._learning_rate_N_max = learning_rate_N_max
        self._learning_rate = tf.placeholder(dtype, shape=[])

        with tf.name_scope("network_variables"):
            # policy parameters
            self._mu_theta = tf.get_variable("mu_theta", [2, num_input],
                                             initializer=tf.random_normal_initializer(stddev=0.3), dtype=dtype)

        self._mu = tf.matmul(self._states, tf.transpose(self._mu_theta))

        self._sigma = tf.placeholder(dtype, shape=[])

        self._optimizer = optimizer(learning_rate=self._learning_rate)

        self._discounted_rewards = tf.placeholder(dtype, [1, 1], name="discounted_rewards")
        self._taken_actions = tf.placeholder(dtype, [1, 2], name="taken_actions")

        # we'll get the policy gradient by using -log(pdf), where pdf is the PDF of the Normal distribution
        self._loss = -tf.log(tf.sqrt(1/(2 * np.pi * self._sigma**2)) * tf.exp(-(self._taken_actions - self._mu)**2/(2 * self._sigma**2))) * self._discounted_rewards

        self._train_op = self._optimizer.minimize(self._loss)

        self._sess.run(tf.global_variables_initializer())

        self._all_rewards = []
        self._max_reward_length = 1000000

        self._sigma_in = init_sigma
        self._num_input = num_input
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
            self._states: np.reshape(system_state, (1, self._num_input)),
            self._sigma: self._sigma_in
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

    def _update_sigma(self, iteration):
        if self._sigma_in <= .1:
            return
        if iteration > 0 and iteration % 1000 == 0:
            self._sigma_in -= .1

    def update_parameters(self, iteration, discounted_return, episode_states):
        episode_actions = self._sensor_actions
        learning_rate = self._gen_learning_rate(iteration, l_max=self._init_learning_rate,
                                                l_min=self._min_learning_rate, N_max=self._learning_rate_N_max)
        self._update_sigma(iteration)

        # reduce gradient variance by normalization
        self._all_rewards += discounted_return.tolist()
        self._all_rewards = self._all_rewards[:self._max_reward_length]
        discounted_return -= np.mean(self._all_rewards)
        discounted_return /= np.std(self._all_rewards)

        N = len(episode_states)

        all_samples = []
        for t in range(N):
            state  = np.reshape(np.array(episode_states[t]), (1, self._num_input))
            action = np.reshape(np.array(episode_actions[t][0]), (1, 2))
            reward = np.reshape(np.array([discounted_return[t]]), (1, 1))
            sample = [state, action, reward]
            all_samples.append(sample)

        for sample in all_samples:
            # prepare inputs
            states = sample[0]
            actions = sample[1]
            rewards = sample[2]

            self._sess.run([self._train_op], feed_dict={
                self._states:             states,
                self._taken_actions:      actions,
                self._discounted_rewards: rewards,
                self._learning_rate:      learning_rate,
                self._sigma:              self._sigma_in
            })
        return True

    def __str__(self):
        return self.__class__.__name__ + "_" + str(self._num_input) + "_" + str(self._init_learning_rate) + "_" + \
               str(self._min_learning_rate) + "_" + str(self._learning_rate_N_max) + "_" + str(self._sigma_in)
