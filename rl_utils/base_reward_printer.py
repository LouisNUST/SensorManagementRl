class BaseRewardPrinter:
    def __init__(self, window=10):
        self._window = window

    def print_reward(self, episode, episode_rewards):
        pass

    def get_window(self):
        return self._window
