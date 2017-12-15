import random

import numpy as np
import tensorflow as tf
from tensorflow.contrib import rnn


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

    def get_sigmas(self):
        return []  # TODO

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

        self._optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
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

    def get_sigmas(self):
        return []  # TODO

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
    def __init__(self, num_input, init_learning_rate=1e-6, min_learning_rate=1e-10, learning_rate_N_max=10000,
                 shuffle=True, batch_size=1):
        self._sess = tf.Session()
        self._states = tf.placeholder(tf.float32, (None, num_input), name="states")

        self._init_learning_rate = init_learning_rate
        self._min_learning_rate = min_learning_rate
        self._learning_rate_N_max = learning_rate_N_max
        self._learning_rate = tf.placeholder(tf.float32, shape=[])

        # policy parameters
        self._mu_theta = tf.get_variable("mu_theta", [2, 200],
                                         initializer=tf.zeros_initializer())
        self._sigma_theta = tf.get_variable("sigma_theta", [2, 200],
                                            initializer=tf.zeros_initializer())

        # neural featurizer parameters
        self._W1 = tf.get_variable("W1", [num_input, 800],
                                   initializer=tf.random_normal_initializer())
        self._b1 = tf.get_variable("b1", [800],
                                   initializer=tf.constant_initializer(0))
        self._h1 = tf.nn.tanh(tf.matmul(self._states, self._W1) + self._b1)
        self._W2 = tf.get_variable("W2", [800, 1200],
                                   initializer=tf.random_normal_initializer(stddev=0.1))
        self._b2 = tf.get_variable("b2", [1200],
                                   initializer=tf.constant_initializer(0))
        # self._phi = tf.matmul(self._h1, self._W2) + self._b2
        self._h2 = tf.nn.tanh(tf.matmul(self._h1, self._W2) + self._b2)

        self._W3 = tf.get_variable("W3", [1200, 200],
                                   initializer=tf.random_normal_initializer(stddev=0.1))
        self._b3 = tf.get_variable("b3", [200],
                                   initializer=tf.constant_initializer(0))
        self._phi = tf.matmul(self._h2, self._W3) + self._b3

        self._mu = tf.matmul(self._phi, tf.transpose(self._mu_theta))
        self._sigma = tf.matmul(self._phi, tf.transpose(self._sigma_theta))
        self._sigma = tf.nn.softplus(self._sigma) + 1e-5

        self._optimizer = tf.train.GradientDescentOptimizer(learning_rate=self._learning_rate)

        self._discounted_rewards = tf.placeholder(tf.float32, (None, 1), name="discounted_rewards")
        self._taken_actions = tf.placeholder(tf.float32, (None, 2), name="taken_actions")

        # we'll get the policy gradient by using -log(pdf), where pdf is the PDF of the Normal distribution
        self._loss = -tf.log(tf.sqrt(1/(2 * np.pi * self._sigma**2)) * tf.exp(-(self._taken_actions - self._mu)**2/(2 * self._sigma**2))) * self._discounted_rewards

        self._train_op = self._optimizer.minimize(self._loss)

        self._sess.run(tf.global_variables_initializer())

        self._num_input = num_input
        self._shuffle = shuffle
        self._batch_size = batch_size
        self.reset_location()

    def reset_location(self):
        self._init_x = 10000 * random.random() - 5000
        self._init_y = 10000 * random.random() - 5000
        self._initial_location = [self._init_x, self._init_y]
        self._historical_location = [self._initial_location]
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

        self._current_location = [new_x, new_y]
        self._historical_location.append(self._current_location)

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
                self._states: states,
                self._taken_actions:      actions,
                self._discounted_rewards: rewards,
                self._learning_rate:      learning_rate
            })
        return True

    def _minibatches(self, samples, batch_size):
        for i in range(0, len(samples), batch_size):
            yield samples[i:i + batch_size]

    def __str__( self ):
        return self.__class__.__name__ + "_" + str(self._num_input) + "_" + str(self._init_learning_rate) + "_" + \
               str(self._min_learning_rate) + "_" + str(self._learning_rate_N_max) + "_" + str(self._shuffle) + "_" + \
               str(self._batch_size)


class TFStochasticPolicyOTPSensor:
    def __init__(self, num_input, init_learning_rate=0.001):
        self._sess = tf.Session()
        self._states = tf.placeholder(tf.float32, (None, num_input), name="states")

        self._init_learning_rate = init_learning_rate
        self._learning_rate = tf.placeholder(tf.float32, shape=[])

        # policy parameters
        self._mu_theta = tf.get_variable("mu_theta", [2, 20],
                                         initializer=tf.zeros_initializer())
        # self._sigma_theta = tf.get_variable("sigma_theta", [2, 20],
        #                                     initializer=tf.zeros_initializer())

        self._mu = tf.matmul(self._states, tf.transpose(self._mu_theta))
        # self._sigma = tf.matmul(self._states, tf.transpose(self._sigma_theta))
        # self._sigma = tf.nn.softplus(self._sigma) + 1e-5
        self._sigma = 1.

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
        # Gaussian policy
        # mu, sigma = self._sess.run([self._mu, self._sigma], feed_dict={
        #     self._states: np.reshape(system_state, (1, self._num_input))
        # })
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
        return []  # TODO

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


class TFRecurrentStochasticPolicyOTPSensor:
    def __init__(self, num_input, learning_rate=0.001, sigma=1, n_hidden=20):
        self._sigma = sigma
        self._sess = tf.Session()
        num_input = 5
        self._states = tf.placeholder(tf.float32, (None, 1, num_input), name="states")

        self._n_hidden = n_hidden

        # policy parameters
        self._mu_theta = tf.get_variable("mu_theta", [2, self._n_hidden],
                                         initializer=tf.zeros_initializer())
        self._sigma_theta = tf.get_variable("sigma_theta", [2, self._n_hidden],
                                            initializer=tf.zeros_initializer())

        # LSTM featurizer
        input_sequence = tf.unstack(self._states, 1, 1)
        self._lstm_cell = rnn.BasicLSTMCell(self._n_hidden, forget_bias=1.0)
        self._rnn_state_in = self._lstm_cell.zero_state(1, tf.float32)
        outputs, self._rnn_state = rnn.static_rnn(self._lstm_cell, input_sequence, dtype=tf.float32, initial_state=self._rnn_state_in)

        self._phi = outputs[-1]

        self._mu = tf.matmul(self._phi, tf.transpose(self._mu_theta))
        self._sigma = tf.matmul(self._phi, tf.transpose(self._sigma_theta))
        self._sigma = tf.nn.softplus(self._sigma) + 1e-5

        self._optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)

        self._discounted_rewards = tf.placeholder(tf.float32, (None,), name="discounted_rewards")
        self._taken_actions = tf.placeholder(tf.float32, (None, 2), name="taken_actions")

        # we'll get the policy gradient by using -log(pdf), where pdf is the PDF of the Normal distribution
        self._loss = -tf.log(tf.sqrt(1/(2 * np.pi * sigma**2)) * tf.exp(-(self._taken_actions - self._mu)**2/(2 * sigma**2)))

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
        self._curr_rnn_state = (np.zeros([1, self._n_hidden]), np.zeros([1, self._n_hidden]))

    def update_location(self, system_state):
        # Gaussian policy
        system_state = self._filter_state(system_state)
        mu, sigma, s = self._sess.run([self._mu, self._sigma, self._rnn_state], feed_dict={
            self._states: np.reshape(system_state, (1, 1, self._num_input)),
            self._rnn_state_in: self._curr_rnn_state
        })
        self._curr_rnn_state = s
        delta = np.random.normal(mu, sigma)

        self._sensor_actions.append(delta)
        new_x = self._current_location[0] + delta[0][0]
        new_y = self._current_location[1] + delta[0][1]

        self._current_location = [new_x, new_y]
        self._historical_location.append(self._current_location)

    def _filter_state(self, state):
        # state is just: est. target x; est. target y; sensor x; sensor y; bearing
        return [state[0], state[1], state[4], state[5], state[6]]

    def get_current_location(self):
        return self._current_location

    def get_weights(self):
        return []  # TODO for now

    def get_sigmas(self):
        return []  # TODO

    def update_parameters(self, iteration, discounted_return, episode_states):
        episode_actions = self._sensor_actions
        # reset LSTM hidden state at the beginning of the episode update
        curr_rnn_state = (np.zeros([1, self._n_hidden]), np.zeros([1, self._n_hidden]))
        N = len(episode_states)
        for t in range(N-1):
            # prepare inputs
            action = np.array(episode_actions[t])
            states  = np.array([[self._filter_state(episode_states[t])]])
            rewards = np.array([discounted_return[t]])

            # perform one update of training
            _, s = self._sess.run([self._train_op, self._rnn_state], feed_dict={
                self._states: states,
                self._taken_actions: action,
                self._discounted_rewards: rewards,
                self._rnn_state_in: curr_rnn_state
            })
            curr_rnn_state = s
        return True


class TFNeuralNetStochasticPolicyStackingOTPSensor:
    def __init__(self, num_input, learning_rate=1e-6, sigma=1):
        self._sigma = sigma

        self._state_dim = 5
        self._num_past_states = 3
        num_input = self._state_dim * (self._num_past_states + 1)

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

        self._optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)

        self._discounted_rewards = tf.placeholder(tf.float32, (None,), name="discounted_rewards")
        self._taken_actions = tf.placeholder(tf.float32, (None, 2), name="taken_actions")

        # we'll get the policy gradient by using -log(pdf), where pdf is the PDF of the Normal distribution
        self._loss = -tf.log(tf.sqrt(1/(2 * np.pi * sigma**2)) * tf.exp(-(self._taken_actions - self._mu)**2/(2 * sigma**2)))

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
        self._init_past_states()

    def _init_past_states(self):
        self._past_states = []
        for i in range(self._num_past_states):
            self._past_states.append([0.] * self._state_dim)

    def update_location(self, system_state):
        system_state = self._prepare_states(system_state)
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

    def get_sigmas(self):
        return []  # TODO

    def _filter_state(self, state):
        # state is just: est. target x; est. target y; sensor x; sensor y; bearing
        return [state[0], state[1], state[4], state[5], state[6]]

    def _prepare_states(self, curr_state):
        curr_state = self._filter_state(curr_state)
        self._past_states.append(curr_state)
        stack_len = self._num_past_states + 1
        return np.concatenate(self._past_states[-stack_len:])

    def update_parameters(self, iteration, discounted_return, episode_states):
        episode_actions = self._sensor_actions
        self._init_past_states()
        N = len(episode_states)
        for t in range(N-1):
            # prepare inputs
            action = np.array(episode_actions[t])
            states  = np.array([self._prepare_states(episode_states[t])])
            rewards = np.array([discounted_return[t]])

            # perform one update of training
            self._sess.run([self._train_op], feed_dict={
                self._states: states,
                self._taken_actions: action,
                self._discounted_rewards: rewards
            })
        return True