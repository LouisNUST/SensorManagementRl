import numpy as np


class RawRewardPrinter:
    def __init__(self):
        pass

    def print_reward(self, episode, episode_rewards):
        print("%s,%s" % (episode, np.sum(episode_rewards)))
