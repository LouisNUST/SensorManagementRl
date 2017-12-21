import numpy as np


class OTPEnvironment:
    def __init__(self, x_min=-50000, x_max=50000, y_min=-50000, y_max=50000, vel_min=-10, vel_max=10, bearing_variance=1):
        self._x_min = x_min
        self._x_max = x_max
        self._y_min = y_min
        self._y_max = y_max
        self._vel_min = vel_min
        self._vel_max = vel_max
        self._bearing_variance = bearing_variance
        self._bearing_measurements = []

    def get_x_min(self):
        return self._x_min

    def get_x_max(self):
        return self._x_max

    def get_y_min(self):
        return self._y_min

    def get_y_max(self):
        return self._y_max

    def get_vel_min(self):
        return self._vel_min

    def get_vel_max(self):
        return self._vel_max

    def bearing_variance(self):
        return self._bearing_variance

    def generate_bearing(self, target_loc, sensor_loc):
        noiseless_bearing = np.arctan2(target_loc[1] - sensor_loc[1], target_loc[0] - sensor_loc[0])
        self._bearing_measurements.append(noiseless_bearing + np.random.normal(0, self._bearing_variance**2))

    def get_last_bearing_measurement(self):
        return self._bearing_measurements[-1]
