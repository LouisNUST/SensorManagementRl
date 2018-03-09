import numpy as np


class RewardByUncertaintyOld:
    def __init__(self, window_size=50, window_lag=10):
        self._window_size = window_size
        self._window_lag = window_lag
        self.reset()

    def reset(self):
        self._uncertainty = []

    def get_reward(self, sensor, target, tracker):
        unnormalized_uncertainty = np.sum(tracker.get_estimation_error_covariance_matrix().diagonal())
        # reward: see if the uncertainty has decayed or if it has gone below a certain value
        self._uncertainty.append((1.0/tracker.get_max_uncertainty()) * unnormalized_uncertainty)
        if len(self._uncertainty) < self._window_size + self._window_lag:
            return 0
        else:
            current_avg = np.mean(self._uncertainty[-self._window_size:])
            prev_avg = np.mean(self._uncertainty[-(self._window_size+self._window_lag):-self._window_lag])
            if current_avg<prev_avg or self._uncertainty[-1]<.1:
                return 1
            else:
                return 0
