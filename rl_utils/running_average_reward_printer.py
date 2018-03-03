import numpy as np


class RunningAverageRewardPrinter:
    def __init__(self, window=100):
        self._simulation_rewards = []
        self._window = window

    def print_reward(self, episode, episode_rewards):
        self._simulation_rewards.append(sum(episode_rewards))
        print("%s,%s" % (episode, np.mean(self._simulation_rewards)))
        if self._window is not None and episode % self._window == 0 and episode > 0:
            self._simulation_rewards = []
