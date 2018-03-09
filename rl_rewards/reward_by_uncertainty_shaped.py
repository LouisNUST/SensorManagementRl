import numpy as np


class RewardByUncertaintyShaped:
    def __init__(self):
        self.reset()

    def reset(self):
        pass

    def get_reward(self, sensor, target, tracker):
        error_trace = np.trace(tracker.get_estimation_error_covariance_matrix()) / 1E9
        return -error_trace
