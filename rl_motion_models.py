import numpy as np


class ConstantVelocityMotionModel:
    def __init__(self, sample_time=1, heading_rate=1E-8):
        self._T = sample_time
        self._H = heading_rate

    def move(self):
        A = np.array([[1, 0, np.sin(self._H * self._T) / self._H, (np.cos(self._H * self._T) - 1) / self._H]
                         , [0, 1, (1 - np.cos(self._H * self._T)) / self._H, np.sin(self._H * self._T) / self._H],
                      [0, 0, np.cos(self._H * self._T), -np.sin(self._H * self._T)],
                      [0, 0, np.sin(self._H * self._T), np.cos(self._H * self._T)]])
        B = np.array([[self._T**2 / 2.0, 0], [0, self._T**2 / 2.0], [self._T, 0], [0, self._T]])

        return A, B
