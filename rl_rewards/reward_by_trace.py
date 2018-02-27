import numpy as np


class RewardByTrace:
    def __init__(self):
        self._uncertainty = []

    def get_reward(self, sensor, tracker):
        error_trace = np.trace(tracker.get_estimation_error_covariance_matrix())
        self._uncertainty.append(error_trace)
        if len(self._uncertainty) == 1:
            return 0
        error_trace_diff = self._uncertainty[-1] - self._uncertainty[-2]
        if error_trace_diff < 0:
            return 1
        elif error_trace_diff == 0:
            return 0
        elif error_trace_diff > 0:
            return -1
