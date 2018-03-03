import numpy as np

from rl_utils import BaseRewardPrinter


class PeriodicAverageRewardPrinter(BaseRewardPrinter):
    def __init__(self, window):
        super().__init__(window)
        self._simulation_rewards = []

    def print_reward(self, episode, episode_rewards):
        self._simulation_rewards.append(sum(episode_rewards))
        if episode % self._window == 0 and episode > 0:
            print("%s,%s" % (episode, np.mean(self._simulation_rewards)))
            self._simulation_rewards = []
