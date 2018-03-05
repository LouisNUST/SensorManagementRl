import numpy as np

from rl_utils import BaseRewardPrinter


class PeriodicAverageRewardPrinter(BaseRewardPrinter):
    def __init__(self, window, format=None):
        """
        :param window: the number of timesteps to average the reward over
        :param format: a format for the printed reward (e.g. "{:.4E}") 
        """
        super().__init__(window)
        self._simulation_rewards = []
        self._format = format

    def print_reward(self, episode, episode_rewards):
        self._simulation_rewards.append(sum(episode_rewards))
        if episode % self._window == 0 and episode > 0:
            mean_reward = np.mean(self._simulation_rewards)
            reward = mean_reward if self._format is None else self._format.format(mean_reward)
            print("%s,%s" % (episode, reward))
            self._simulation_rewards = []
