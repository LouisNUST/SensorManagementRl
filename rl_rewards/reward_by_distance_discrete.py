import numpy as np


class RewardByDistanceDiscrete:
    def __init__(self):
        self.reset()

    def reset(self):
        self._previous_distance = None

    def get_reward(self, sensor, target, tracker):
        target_location_estimate = target.get_current_location() #tracker.get_target_state_estimate()[0:2].reshape(2)
        sensor_location = sensor.get_current_location()
        distance = np.linalg.norm(np.array(target_location_estimate) - np.array(sensor_location))

        if self._previous_distance is None:
            self._previous_distance = distance
            return 0
        diff = distance - self._previous_distance
        self._previous_distance = distance
        if diff < 0:
            return 1
        elif diff == 0:
            return 0
        elif diff > 0:
            return -1
