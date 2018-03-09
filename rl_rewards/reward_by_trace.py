import numpy as np


class RewardByTrace:
    def __init__(self):
        self.reset()

    def reset(self):
        self._previous_uncertainty = None

    def get_reward(self, sensor, target, tracker):
        current_uncertainty = np.trace(tracker.get_estimation_error_covariance_matrix())
        if self._previous_uncertainty is None:
            self._previous_uncertainty = current_uncertainty
            return 0
        uncertainty_diff = current_uncertainty - self._previous_uncertainty
        self._previous_uncertainty = current_uncertainty
        if uncertainty_diff < 0:
            return 1
        elif uncertainty_diff == 0:
            return 0
        elif uncertainty_diff > 0:
            return -1
