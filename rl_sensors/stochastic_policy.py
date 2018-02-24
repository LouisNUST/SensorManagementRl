import numpy as np


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
