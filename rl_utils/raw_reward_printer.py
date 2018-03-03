import numpy as np

from rl_utils import BaseRewardPrinter


class RawRewardPrinter(BaseRewardPrinter):
    def __init__(self):
        super().__init__()

    def print_reward(self, episode, episode_rewards):
        print("%s,%s" % (episode, np.sum(episode_rewards)))
