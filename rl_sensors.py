import random

import numpy as np
import tensorflow as tf


class StochasticPolicyOTPSensor:
    def __init__(self, num_features, parameter_updater, sigma=1):
        self._weights = np.random.normal(0, 1, [2, num_features])
        self._sigma = sigma
        self._num_features = num_features
        self._parameter_updater = parameter_updater
        self.reset_location()

    def reset_location(self):
        self._init_x = 10000 * random.random() - 5000
        self._init_y = 10000 * random.random() - 5000
        self._initial_location = [self._init_x, self._init_y]
        self._historical_location = [self._initial_location]
        self._current_location = self._initial_location
        self._sensor_actions = []

    def update_location(self, featurized_system_state):
        # Gaussian policy
        delta = np.random.normal(self._weights.dot(featurized_system_state), self._sigma)
        self._sensor_actions.append(delta)
        new_x = self._current_location[0] + delta[0]
        new_y = self._current_location[1] + delta[1]

        self._current_location = [new_x, new_y]
        self._historical_location.append(self._current_location)

    def get_current_location(self):
        return self._current_location

    def get_weights(self):
        return self._weights

    def update_parameters(self, iteration, discounted_return, episode_states):
        condition, new_weights = self._parameter_updater.update_parameters(self._weights, iteration,
                                                                           self._sensor_actions, episode_states,
                                                                           discounted_return, self._num_features,
                                                                           self._sigma)
        self._weights = new_weights
        return condition


class TFNeuralNetDeterministicPolicyOTPSensor:
    def __init__(self, num_input, learning_rate=0.001):
        self._sess = tf.Session()
        self._states = tf.placeholder(tf.float32, (None, num_input), name="states")
        self._W1 = tf.get_variable("W1", [num_input, 20],
                                   initializer=tf.random_normal_initializer())
        self._b1 = tf.get_variable("b1", [20],
                                   initializer=tf.constant_initializer(0))
        self._h1 = self.leaky_relu(tf.matmul(self._states, self._W1) + self._b1, alpha=0.3)
        self._W2 = tf.get_variable("W2", [20, 20],
                                   initializer=tf.random_normal_initializer(stddev=0.1))
        self._b2 = tf.get_variable("b2", [20],
                                   initializer=tf.constant_initializer(0))
        self._h2 = self.leaky_relu(tf.matmul(self._h1, self._W2) + self._b2, alpha=0.3)

        self._W3 = tf.get_variable("W3", [20, 2],
                                   initializer=tf.random_normal_initializer(stddev=0.1))
        self._b3 = tf.get_variable("b3", [2],
                                   initializer=tf.constant_initializer(0))
        self._out = tf.matmul(self._h2, self._W3) + self._b3

        self._optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate, decay=0.9)
        self._discounted_rewards = tf.placeholder(tf.float32, (None,), name="discounted_rewards")

        # the policy gradient is: grad log pi(s);
        #   we use -log pi(s) here because we want to maximize J, but we're doing minimization here
        self._score = -tf.log(tf.clip_by_value(self._out, 1e-5, 1.0)) * self._discounted_rewards

        self._loss = tf.reduce_sum(self._score)

        self._train_op = self._optimizer.minimize(self._loss)

        self._sess.run(tf.global_variables_initializer())

        self._num_input = num_input
        self.reset_location()

    def leaky_relu(self, x, alpha):
        return tf.nn.relu(x) - alpha * tf.nn.relu(-x)

    def reset_location(self):
        self._init_x = 10000 * random.random() - 5000
        self._init_y = 10000 * random.random() - 5000
        self._initial_location = [self._init_x, self._init_y]
        self._historical_location = [self._initial_location]
        self._current_location = self._initial_location
        self._sensor_actions = []

    def update_location(self, system_state):
        # deterministic policy
        delta = self._sess.run(self._out, feed_dict={self._states: np.reshape(system_state, (1, self._num_input))})

        self._sensor_actions.append(delta)
        new_x = self._current_location[0] + delta[0][0]
        new_y = self._current_location[1] + delta[0][1]

        self._current_location = [new_x, new_y]
        self._historical_location.append(self._current_location)

    def get_current_location(self):
        return self._current_location

    def get_weights(self):
        return []  # TODO for now

    def update_parameters(self, iteration, discounted_return, episode_states):
        N = len(episode_states)
        for t in range(N-1):
            # prepare inputs
            states  = np.array([episode_states[t]])
            rewards = np.array([discounted_return[t]])

            # perform one update of training
            self._sess.run([self._train_op], feed_dict={
                self._states: states,
                self._discounted_rewards: rewards
            })
        return True


class TFNeuralNetStochasticPolicyOTPSensor:
    def __init__(self, num_input, learning_rate=0.001, sigma=1):
        self._sigma = sigma
        self._sess = tf.Session()
        self._states = tf.placeholder(tf.float32, (None, num_input), name="states")

        # policy parameters
        self._policy_params = tf.get_variable("theta", [2, 20],
                                   initializer=tf.random_normal_initializer())

        # neural featurizer parameters
        self._W1 = tf.get_variable("W1", [num_input, 20],
                                   initializer=tf.random_normal_initializer())
        self._b1 = tf.get_variable("b1", [20],
                                   initializer=tf.constant_initializer(0))
        self._h1 = tf.nn.tanh(tf.matmul(self._states, self._W1) + self._b1)
        self._W2 = tf.get_variable("W2", [20, 20],
                                   initializer=tf.random_normal_initializer(stddev=0.1))
        self._b2 = tf.get_variable("b2", [20],
                                   initializer=tf.constant_initializer(0))
        # self._phi = tf.matmul(self._h1, self._W2) + self._b2
        self._h2 = tf.nn.tanh(tf.matmul(self._h1, self._W2) + self._b2)

        self._W3 = tf.get_variable("W3", [20, 20],
                                   initializer=tf.random_normal_initializer(stddev=0.1))
        self._b3 = tf.get_variable("b3", [20],
                                   initializer=tf.constant_initializer(0))
        self._phi = tf.matmul(self._h2, self._W3) + self._b3

        self._mu = tf.matmul(self._phi, tf.transpose(self._policy_params))

        self._optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate, decay=0.9)

        self._discounted_rewards = tf.placeholder(tf.float32, (None,), name="discounted_rewards")
        self._taken_actions = tf.placeholder(tf.float32, (None, 2), name="taken_actions")

        # we'll get the policy gradient by using -log(pdf), where pdf is the PDF of the Normal distribution
        # self._score = -tf.log(tf.sqrt(1/(2 * np.pi * sigma**2)) * tf.exp(-(self._taken_actions - self._mu)**2/(2 * sigma**2)))
        normal_dist = tf.contrib.distributions.Normal(self._mu, [1.])
        self._score = -normal_dist.log_prob(self._taken_actions)

        self._loss = tf.reduce_mean(self._score)

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
        # Gaussian policy
        mu = self._sess.run(self._mu, feed_dict={self._states: np.reshape(system_state, (1, self._num_input))})
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

    def update_parameters(self, iteration, discounted_return, episode_states):
        episode_actions = self._sensor_actions
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
                self._discounted_rewards: rewards
            })
        return True


class TFStochasticPolicyOTPSensor:
    def __init__(self, num_input, learning_rate=0.001, sigma=1):
        self._sigma = sigma
        self._sess = tf.Session()
        self._states = tf.placeholder(tf.float32, (None, num_input), name="states")

        # policy parameters
        self._policy_params = tf.get_variable("theta", [2, 20],
                                              initializer=tf.random_normal_initializer())

        self._mu = tf.matmul(self._states, tf.transpose(self._policy_params))

        self._optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate, decay=0.9)

        self._discounted_rewards = tf.placeholder(tf.float32, (None,), name="discounted_rewards")
        self._taken_actions = tf.placeholder(tf.float32, (None, 2), name="taken_actions")

        # we'll get the policy gradient by using -log(pdf), where pdf is the PDF of the Normal distribution
        # self._score = -tf.log(tf.sqrt(1/(2 * np.pi * sigma**2)) * tf.exp(-(self._taken_actions - self._mu)**2/(2 * sigma**2)))
        normal_dist = tf.contrib.distributions.Normal(self._mu, [1.])
        self._score = -normal_dist.log_prob(self._taken_actions)

        self._loss = tf.reduce_mean(self._score)

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
        # Gaussian policy
        mu = self._sess.run(self._mu, feed_dict={self._states: np.reshape(system_state, (1, self._num_input))})
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

    def update_parameters(self, iteration, discounted_return, episode_states):
        episode_actions = self._sensor_actions
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
                self._discounted_rewards: rewards
            })
        return True
