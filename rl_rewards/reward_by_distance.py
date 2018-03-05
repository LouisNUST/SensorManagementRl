import numpy as np


class RewardByDistance:
    def __init__(self):
        pass

    def get_reward(self, sensor, target, tracker):
        target_location_estimate = target.get_current_location() #tracker.get_target_state_estimate()[0:2].reshape(2)
        sensor_location = sensor.get_current_location()
        distance = np.linalg.norm(np.array(target_location_estimate) - np.array(sensor_location))
        return -distance
