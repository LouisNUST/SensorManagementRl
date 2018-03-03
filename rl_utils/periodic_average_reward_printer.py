import numpy as np


class PeriodicAverageRewardPrinter:
    def __init__(self, window=100):
        self._simulation_rewards = []
        self._window = window

    def print_reward(self, episode, episode_rewards):
        self._simulation_rewards.append(sum(episode_rewards))
        if episode % self._window == 0 and episode > 0:
            print("%s,%s" % (episode, np.mean(self._simulation_rewards)))
            self._simulation_rewards = []
