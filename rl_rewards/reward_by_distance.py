import numpy as np


class RewardByDistance:
    def __init__(self):
        pass

    def get_reward(self, sensor, tracker):
        target_location_estimate = tracker.get_target_state_estimate()[0:2].reshape(2)
        sensor_location = sensor.get_current_location()
        distance = np.linalg.norm(target_location_estimate - sensor_location)
        return -distance
