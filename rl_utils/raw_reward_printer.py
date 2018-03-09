import numpy as np

from rl_utils import BaseRewardPrinter


class RawRewardPrinter(BaseRewardPrinter):
    def __init__(self, format=None):
        super().__init__()
        self._format = format

    def print_reward(self, episode, episode_rewards):
        total_reward = np.sum(episode_rewards)
        reward = total_reward if self._format is None else self._format.format(total_reward)
        print("%s,%s" % (episode, reward))
