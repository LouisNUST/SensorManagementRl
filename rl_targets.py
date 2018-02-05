from rl_motion_models import ConstantVelocityMotionModel
import random
import numpy as np


class ConstantVelocityTarget:
    def __init__(self, x_variance=0, y_variance=0, init_pos=None, init_vel=None):
        self._motion_model = ConstantVelocityMotionModel()
        # initialize target location (x, y) and velocity (x_dot, y_dot)
        if init_pos is None:
            self._x = 2000 * random.random() - 1000
            self._y = 2000 * random.random() - 1000
        else:
            self._x = init_pos[0]
            self._y = init_pos[1]
        if init_vel is None:
            self._x_dot = 10 * random.random() - 5
            self._y_dot = 10 * random.random() - 5
        else:
            self._x_dot = init_vel[0]
            self._y_dot = init_vel[1]
        self._initial_location = [self._x, self._y]
        self._current_location = self._initial_location
        self._historical_location = [self._initial_location]
        self._initial_velocity = [self._x_dot, self._y_dot]
        self._current_velocity = self._initial_velocity
        self._historical_velocity = [self._initial_velocity]
        self._x_variance = x_variance
        self._y_variance = y_variance

    def move(self):
        return self._motion_model.move()

    def get_x(self):
        return self._x

    def get_y(self):
        return self._y

    def get_x_variance(self):
        return self._x_variance

    def get_y_variance(self):
        return self._y_variance

    def update_location(self):

        A, B = self.move()

        noise_x = np.random.normal(0, self._x_variance)
        noise_y = np.random.normal(0, self._y_variance)

        current_state = [self._current_location[0], self._current_location[1], self._current_velocity[0], self._current_velocity[1]]
        new_state = A.dot(current_state) + B.dot(np.array([noise_x, noise_y]))  # This is the new state

        new_location = [new_state[0], new_state[1]]
        self._current_location = new_location
        self._historical_location.append(self._current_location)

        new_velocity = [new_state[2], new_state[3]]
        self._current_velocity = new_velocity
        self._historical_velocity.append(self._current_velocity)

    def get_current_location(self):
        return self._current_location

    def get_current_velocity(self):
        return self._current_velocity
