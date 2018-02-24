import random

import numpy as np
import tensorflow as tf


class TFStochasticPolicyOTPSensor:
    def __init__(self, num_input, init_learning_rate=0.001):
        self._sess = tf.Session()
        self._states = tf.placeholder(tf.float32, (None, num_input), name="states")

        self._init_learning_rate = init_learning_rate
        self._learning_rate = tf.placeholder(tf.float32, shape=[])

        # policy parameters
        self._mu_theta = tf.get_variable("mu_theta", [2, 20],
                                         initializer=tf.zeros_initializer())

        self._mu = tf.matmul(self._states, tf.transpose(self._mu_theta))
        self._sigma = 1

        self._optimizer = tf.train.GradientDescentOptimizer(learning_rate=self._learning_rate)

        self._discounted_rewards = tf.placeholder(tf.float32, (None,), name="discounted_rewards")
        self._taken_actions = tf.placeholder(tf.float32, (None, 2), name="taken_actions")

        # we'll get the policy gradient by using -log(pdf), where pdf is the PDF of the Normal distribution
        self._loss = -tf.log(tf.sqrt(1/(2 * np.pi * self._sigma**2)) * tf.exp(-(self._taken_actions - self._mu)**2/(2 * self._sigma**2)))

        self._gradients = self._optimizer.compute_gradients(self._loss)
        for i, (grad, var) in enumerate(self._gradients):
            if grad is not None:
                self._gradients[i] = (grad * self._discounted_rewards, var)
        self._train_op = self._optimizer.apply_gradients(self._gradients)

        self._sess.run(tf.global_variables_initializer())

        self._num_input = num_input
        self.reset_location()

    def reset_location(self):
        self._init_x = 10000 * random.random() - 5000
        self._init_y = 10000 * random.random() - 5000
        self._initial_location = [self._init_x, self._init_y]
        self._historical_location = [self._initial_location]
        self._current_location = self._initial_location
        self._sensor_actions = []

    def update_location(self, system_state):
        mu = self._sess.run(self._mu, feed_dict={
            self._states: np.reshape(system_state, (1, self._num_input))
        })
        delta = np.random.normal(mu, self._sigma)

        self._sensor_actions.append(delta)
        new_x = self._current_location[0] + delta[0][0]
        new_y = self._current_location[1] + delta[0][1]

        self._current_location = [new_x, new_y]
        self._historical_location.append(self._current_location)

    def get_current_location(self):
        return self._current_location

    def get_weights(self):
        return []  # TODO for now

    def get_sigmas(self):
        return [[0., 0.]] * len(self._sensor_actions)  # TODO

    def _gen_learning_rate(self, iteration, l_max, l_min, N_max):
        if iteration > N_max:
            return l_min
        alpha = 2 * l_max
        beta = np.log((alpha / l_min - 1)) / N_max
        return alpha / (1 + np.exp(beta * iteration))

    def update_parameters(self, iteration, discounted_return, episode_states):
        episode_actions = self._sensor_actions
        learning_rate = self._gen_learning_rate(iteration, l_max=self._init_learning_rate, l_min=1E-8, N_max=10000)
        N = len(episode_states)
        for t in range(N-1):
            # prepare inputs
            actions = np.array(episode_actions[t])
            states  = np.array([episode_states[t]])
            rewards = np.array([discounted_return[t]])

            # perform one update of training
            self._sess.run([self._train_op], feed_dict={
                self._states: states,
                self._taken_actions:      actions,
                self._discounted_rewards: rewards,
                self._learning_rate:      learning_rate
            })
        return True

    def __str__(self):
        return self.__class__.__name__ + "_" + str(self._num_input) + "_" + str(self._sigma) + "_" + \
               str(self._init_learning_rate)
