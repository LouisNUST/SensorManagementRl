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

        self._all_rewards = []
        self._max_reward_length = 1000000

        self.reset_location()

    def reset_location(self):
        self._init_x = 2000 #10000 * random.random() - 5000
        self._init_y = 0 #10000 * random.random() - 5000
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
        return [[0., 0.]] * len(self._sensor_actions)

    def update_parameters(self, iteration, discounted_return, episode_states):

        # r = 0
        # discounted_return = np.zeros(len(rewards))
        # for t in reversed(range(len(rewards))):
        #     r = rewards[t] + 0.99 * r
        #     discounted_return[t] = r

        # reduce gradient variance by normalization
        self._all_rewards += discounted_return.tolist()
        self._all_rewards = self._all_rewards[:self._max_reward_length]
        discounted_return -= np.mean(self._all_rewards)
        discounted_return /= np.std(self._all_rewards)

        condition, new_weights = self._parameter_updater.update_parameters(self._weights, iteration,
                                                                           self._sensor_actions, episode_states,
                                                                           discounted_return, self._num_features,
                                                                           self._sigma)
        self._weights = new_weights
        return condition

    def __str__(self):
        return self.__class__.__name__ + "_" + str(self._num_features) + "_" + str(self._sigma) + "_" + \
               str(self._parameter_updater._learning_rate)


class RandomPolicyOTPSensor:
    def __init__(self, init_pos=None):
        self._init_pos = init_pos

    def reset_location(self):
        if self._init_pos is None:
            self._init_x = 10000 * random.random() - 5000
            self._init_y = 10000 * random.random() - 5000
        else:
            self._init_x = self._init_pos[0]
            self._init_y = self._init_pos[1]
        self._initial_location = [self._init_x, self._init_y]
        self._current_location = self._initial_location
        self._sensor_sigmas = []

    def update_location(self, system_state):
        delta = np.random.normal([0., 0.], 1)

        self._sensor_sigmas.append(1)
        new_x = self._current_location[0] + delta[0]
        new_y = self._current_location[1] + delta[1]
        new_x = np.clip(new_x, -50000, 50000)
        new_y = np.clip(new_y, -50000, 50000)
        self._current_location = [new_x, new_y]

    def get_current_location(self):
        return self._current_location

    def get_weights(self):
        return []  # TODO for now

    def get_sigmas(self):
        return self._sensor_sigmas

    def update_parameters(self, iteration, discounted_return, episode_states):
        return True

    def __str__(self):
        return self.__class__.__name__


class TFNeuralNetStochasticPolicyOTPSensor:
    def __init__(self, num_input, init_learning_rate=1e-6, min_learning_rate=1e-10, learning_rate_N_max=10000,
                 sigma=None, shuffle=True, batch_size=1, init_pos=None, value_learning_rate=1e-6):
        self._sess = tf.Session()
        self._states = tf.placeholder(tf.float64, (None, num_input), name="states")

        self._init_learning_rate = init_learning_rate
        self._min_learning_rate = min_learning_rate
        self._learning_rate_N_max = learning_rate_N_max
        self._learning_rate = tf.placeholder(tf.float64, shape=[])

        self._mu_theta_hidden = 800
        self._sigma_theta_hidden = 100
        self._layer1_hidden = 1600
        self._layer2_hidden = self._mu_theta_hidden

        # policy parameters
        self._mu_theta = tf.get_variable("mu_theta", [2, self._mu_theta_hidden],
                                         initializer=tf.zeros_initializer(), dtype=tf.float64)

        if sigma is None:
            self._sigma_theta = tf.get_variable("sigma_theta", [2, self._sigma_theta_hidden],
                                                initializer=tf.zeros_initializer(), dtype=tf.float64)

        # value parameters
        self._W_baseline = tf.get_variable("W_baseline", [num_input, 800],
                                           initializer=tf.random_normal_initializer(stddev=0.1), dtype=tf.float64)
        self._b_baseline = tf.get_variable("b_baseline", [800],
                                   initializer=tf.constant_initializer(0), dtype=tf.float64)
        self._h_baseline = tf.nn.tanh(tf.matmul(self._states, self._W_baseline) + self._b_baseline)

        self._W2_baseline = tf.get_variable("W2_baseline", [800, 1600],
                                           initializer=tf.random_normal_initializer(stddev=0.1), dtype=tf.float64)
        self._b2_baseline = tf.get_variable("b2_baseline", [1600],
                                           initializer=tf.constant_initializer(0), dtype=tf.float64)
        self._h2_baseline = tf.nn.tanh(tf.matmul(self._h_baseline, self._W2_baseline) + self._b2_baseline)

        self._W_out_baseline = tf.get_variable("W_out_baseline", [1600, 1],
                                              initializer=tf.random_normal_initializer(), dtype=tf.float64)
        self._b_out_baseline = tf.get_variable("b_out_baseline", [1],
                                           initializer=tf.constant_initializer(0), dtype=tf.float64)
        self._predicted_baseline = tf.matmul(self._h2_baseline, self._W_out_baseline) + self._b_out_baseline

        # neural featurizer parameters
        self._W1 = tf.get_variable("W1", [num_input, self._layer1_hidden],
                                   initializer=tf.random_normal_initializer(), dtype=tf.float64)
        self._b1 = tf.get_variable("b1", [self._layer1_hidden],
                                   initializer=tf.constant_initializer(0), dtype=tf.float64)
        self._h1 = tf.nn.tanh(tf.matmul(self._states, self._W1) + self._b1)
        self._W2 = tf.get_variable("W2", [self._layer1_hidden, self._layer2_hidden],
                                   initializer=tf.random_normal_initializer(stddev=0.1), dtype=tf.float64)
        self._b2 = tf.get_variable("b2", [self._layer2_hidden],
                                   initializer=tf.constant_initializer(0), dtype=tf.float64)
        self._phi = tf.nn.tanh(tf.matmul(self._h1, self._W2) + self._b2)

        self._mu = tf.matmul(self._phi, tf.transpose(self._mu_theta))

        if sigma is None:
            self._sigma = tf.reduce_sum(self._sigma_theta, 1)
            self._sigma = tf.reshape(tf.exp(self._sigma), [1, 2])
        else:
            self._sigma = tf.constant(sigma, dtype=tf.float64)

        self._optimizer = tf.train.GradientDescentOptimizer(learning_rate=self._learning_rate)
        # self._optimizer = tf.train.AdamOptimizer(learning_rate=init_learning_rate)
        # self._optimizer = tf.train.RMSPropOptimizer(learning_rate=init_learning_rate, decay=0.9)

        self._value_optimizer = tf.train.GradientDescentOptimizer(learning_rate=value_learning_rate)

        self._discounted_rewards = tf.placeholder(tf.float64, (None, 1), name="discounted_rewards")
        self._taken_actions = tf.placeholder(tf.float64, (None, 2), name="taken_actions")
        self._baselines = tf.placeholder(tf.float64, (None, 1), name="baseline")

        # we'll get the policy gradient by using -log(pdf), where pdf is the PDF of the Normal distribution
        self._loss = -tf.log(tf.sqrt(1/(2 * np.pi * self._sigma**2)) * tf.exp(-(self._taken_actions - self._mu)**2/(2 * self._sigma**2))) * (self._discounted_rewards - self._baselines)
        self._train_op = self._optimizer.minimize(self._loss)

        self._value_loss = tf.losses.mean_squared_error(self._discounted_rewards, self._predicted_baseline)
        self._value_train_op = self._value_optimizer.minimize(self._value_loss)

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

        baselines = []
        for t in range(N-1):
            baseline = self._sess.run([self._predicted_baseline], feed_dict={
                self._states: np.reshape(np.array(episode_states[t]), (1, self._num_input))
            })
            baselines.append(baseline)

        all_samples = []
        for t in range(N-1):
            state  = np.reshape(np.array(episode_states[t]), self._num_input)
            action = episode_actions[t][0]
            reward = [discounted_return[t]]
            baseline = baselines[t][0][0]
            sample = [state, action, reward, baseline]
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
            baselines = [row[3] for row in batch]

            self._sess.run([self._train_op], feed_dict={
                self._states: states,
                self._taken_actions:      actions,
                self._discounted_rewards: rewards,
                self._learning_rate:      learning_rate,
                self._baselines:          baselines
            })

        value_losses = []
        for b in range(len(batches)):
            batch = batches[b]
            states = [row[0] for row in batch]
            rewards = [row[2] for row in batch]
            _, loss = self._sess.run([self._value_train_op, self._value_loss], feed_dict={
                self._states: states,
                self._discounted_rewards: rewards
            })
            value_losses.append(loss)
        print("value loss %s" % np.mean(value_losses))

        return True

    def _minibatches(self, samples, batch_size):
        for i in range(0, len(samples), batch_size):
            yield samples[i:i + batch_size]

    def __str__(self):
        return self.__class__.__name__ + "_" + str(self._num_input) + "_" + str(self._init_learning_rate) + "_" + \
               str(self._min_learning_rate) + "_" + str(self._learning_rate_N_max) + "_" + str(self._sigma_in) + "_" + \
               str(self._shuffle) + "_" + str(self._batch_size)


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


class TFStochasticPolicyWithSigmaOTPSensor:
    def __init__(self, num_input, init_learning_rate=0.001):
        self._sess = tf.Session()
        self._states = tf.placeholder(tf.float32, (None, num_input), name="states")

        self._init_learning_rate = init_learning_rate
        self._learning_rate = tf.placeholder(tf.float32, shape=[])

        # policy parameters
        self._mu_theta = tf.get_variable("mu_theta", [2, 20],
                                         initializer=tf.zeros_initializer())
        self._sigma_theta = tf.get_variable("sigma_theta", [2, 20],
                                            initializer=tf.zeros_initializer())

        self._mu = tf.matmul(self._states, tf.transpose(self._mu_theta))
        self._sigma = tf.matmul(self._states, tf.transpose(self._sigma_theta))
        self._sigma = tf.nn.softplus(self._sigma) + 1e-5

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
        return self.__class__.__name__ + "_" + str(self._num_input) + "_" + str(self._init_learning_rate)


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
        self._sensor_sigmas = []

    def update_location(self, system_state):
        # Gaussian policy
        system_state = self._filter_state(system_state)
        mu, sigma, s = self._sess.run([self._mu, self._sigma, self._rnn_state], feed_dict={
            self._states: np.reshape(system_state, (1, 1, self._num_input)),
            self._rnn_state_in: self._curr_rnn_state
        })
        self._curr_rnn_state = s
        delta = np.random.normal(mu, sigma)

        self._sensor_sigmas.append(sigma)
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
        return self._sensor_sigmas

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
    def __init__(self, state_dim=3, num_states=5, init_learning_rate=1e-6, min_learning_rate=1e-10,
                 learning_rate_N_max=10000, shuffle=True, batch_size=1):

        self._init_learning_rate = init_learning_rate
        self._min_learning_rate = min_learning_rate
        self._learning_rate_N_max = learning_rate_N_max
        self._learning_rate = tf.placeholder(tf.float32, shape=[])

        self._state_dim = state_dim
        self._num_states = num_states
        num_input = self._state_dim * self._num_states

        self._sess = tf.Session()
        self._states = tf.placeholder(tf.float32, (None, num_input), name="states")

        # policy parameters
        self._mu_theta = tf.get_variable("mu_theta", [2, 200],
                                         initializer=tf.random_normal_initializer())
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
        self._init_past_states()

    def _init_past_states(self):
        self._past_states = []
        for i in range(self._num_states - 1):
            self._past_states.append([0.] * self._state_dim)

    def update_location(self, system_state):
        system_state = self._prepare_states(system_state)
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

    def _filter_state(self, state):
        if self._state_dim == 3:
            # state is just: sensor x; sensor y; bearing
            return [state[4], state[5], state[6]]
        elif self._state_dim == 5:
            # state is just: est. target x; est. target y; sensor x; sensor y; bearing
            return [state[0], state[1], state[4], state[5], state[6]]
        raise Exception("state dimension not supported: %s" % self._state_dim)

    def _prepare_states(self, curr_state):
        curr_state = self._filter_state(curr_state)
        self._past_states.append(curr_state)
        stack_len = self._num_states
        return np.concatenate(self._past_states[-stack_len:])

    def _minibatches(self, samples, batch_size):
        for i in range(0, len(samples), batch_size):
            yield samples[i:i + batch_size]

    def update_parameters(self, iteration, discounted_return, episode_states):
        episode_actions = self._sensor_actions
        self._init_past_states()
        learning_rate = self._gen_learning_rate(iteration, l_max=self._init_learning_rate,
                                                l_min=self._min_learning_rate, N_max=self._learning_rate_N_max)

        N = len(episode_states)

        all_samples = []
        for t in range(N-1):
            state  = np.reshape(np.array(self._prepare_states(episode_states[t])), self._num_input)
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

    def __str__(self):
        return self.__class__.__name__ + "_" + str(self._state_dim) + "_" + str(self._num_states) + "_" + \
               str(self._init_learning_rate) + "_" + str(self._min_learning_rate) + "_" + \
               str(self._learning_rate_N_max) + "_" + str(self._shuffle) + "_" + str(self._batch_size)