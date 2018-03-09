import numpy as np


class RewardByUncertainty:
    def __init__(self, window_size=50, window_lag=10):
        self._window_size = window_size
        self._window_lag = window_lag
        self.reset()

    def reset(self):
        self._uncertainty = []

    def _linear_lsq(self, batch):
        x = np.array(range(0, len(batch)))
        A = np.vstack([x, np.ones(len(x))]).T
        m, c = np.linalg.lstsq(A, batch)[0]
        return m, c

    def get_reward(self, sensor, target, tracker):
        unnormalized_uncertainty = np.sum(tracker.get_estimation_error_covariance_matrix().diagonal())
        # reward: see if the uncertainty has decayed or if it has gone below a certain value
        self._uncertainty.append((1.0/tracker.get_max_uncertainty()) * unnormalized_uncertainty)
        if len(self._uncertainty) < self._window_size + self._window_lag:
            return 0
        else:
            slope, c = self._linear_lsq(self._uncertainty[-100:])
            if self._uncertainty[-1] < self._uncertainty[-2] or (slope < 1E-4 and c < .1):
                return 1
            else:
                return 0
