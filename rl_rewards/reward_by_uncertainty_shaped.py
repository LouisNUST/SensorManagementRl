import numpy as np


class RewardByUncertaintyShaped:
    def __init__(self):
        pass

    def get_reward(self, sensor, target, tracker):
        error_trace = np.trace(tracker.get_estimation_error_covariance_matrix())
        return -error_trace
