import numpy as np


class EKFTracker:
    def __init__(self, target_x, target_y, max_uncertainty, A, B, x_var, y_var, bearing_var):

        # initial covariance of state estimation
        init_covariance = np.diag([max_uncertainty, max_uncertainty, max_uncertainty, max_uncertainty])

        # init state for the tracker (tracker doesn't know about the initial state)
        init_estimate = [target_x + np.random.normal(0, 5), target_y + np.random.normal(0, 5), np.random.normal(0, 5), np.random.normal(0, 5)]

        self._init_estimate = init_estimate
        self._init_covariance = init_covariance
        self._bearing_var = bearing_var
        self._A = A
        self._B = B
        self._x_var = x_var
        self._y_var = y_var

        self._max_uncertainty = max_uncertainty

        self._x_k_k = np.array(init_estimate).reshape(len(init_estimate), 1)
        self._x_k_km1 = self._x_k_k
        self._p_k_k = init_covariance
        self._p_k_km1 = init_covariance
        self._S_k = 1E-5
        self._meas_vec = []

        self._innovation_list = []
        self._innovation_var = []
        self._gain = []

    def get_target_state_estimate(self):
        return self._x_k_k

    def get_estimation_error_covariance_matrix(self):
        return self._p_k_k

    def get_predicted_estimation_error_covariance_matrix(self):
        return self._p_k_km1

    def get_max_uncertainty(self):
        return self._max_uncertainty

    def get_linearized_measurment_vector(self,target_state,sensor_state):
        relative_location = target_state[0:2] - np.array(sensor_state[0:2]).reshape(2,1)  ##[x-x_s,y-y_s]
        measurement_vector = np.array([-relative_location[1] / ((np.linalg.norm(relative_location)) ** 2),
                                       relative_location[0] / ((np.linalg.norm(relative_location)) ** 2), [0], [0]])
        measurement_vector = measurement_vector.transpose()
        return (measurement_vector)

    def linearized_predicted_measurement(self,sensor_state):
        sensor_state = np.array(sensor_state).reshape(len(sensor_state),1)
        measurement_vector = self.get_linearized_measurment_vector(self._x_k_km1, sensor_state)#Linearize the measurement model
        #predicted_measurement = measurement_vector.dot(np.array(self.x_k_km1))
        predicted_measurement =  np.arctan2(self._x_k_km1[1] - sensor_state[1], self._x_k_km1[0] - sensor_state[0])
        if predicted_measurement<0:predicted_measurement+= 2*np.pi
        return (predicted_measurement,measurement_vector)

    def predicted_state(self,sensor_state,measurement):

        Q = np.eye(2)
        Q[0,0] = .1
        Q[1,1] = .1

        #Q[0,0] = 5
        #Q[1,1] = 5
        predicted_noise_covariance = (self._B.dot(Q)).dot(self._B.transpose())
        self._x_k_km1 = self._A.dot(self._x_k_k)
        self._p_k_km1 = (self._A.dot(self._p_k_k)).dot(self._A.transpose()) + predicted_noise_covariance
        predicted_measurement, measurement_vector = self.linearized_predicted_measurement(sensor_state)
        self._meas_vec.append(measurement_vector)
        #measurement_vector = measurement_vector.reshape(1,len(measurement_vector))
        self._S_k = (measurement_vector.dot(self._p_k_km1)).dot(measurement_vector.transpose()) + self._bearing_var
        self._innovation_list.append(measurement - predicted_measurement)
        self._innovation_var.append(self._S_k)

    def update_states(self,sensor_state,measurement):
        self.predicted_state(sensor_state,measurement)#prediction-phase
        measurement_vector = self.get_linearized_measurment_vector(self._x_k_km1, sensor_state)  # Linearize the measurement model
        #calculate Kalman gain
        kalman_gain = (self._p_k_km1.dot(measurement_vector.transpose())) / self._S_k

        self._x_k_k = self._x_k_km1 + kalman_gain * self._innovation_list[-1]
        self._p_k_k = self._p_k_km1 - (kalman_gain.dot(measurement_vector)).dot(self._p_k_km1)
        self._gain.append(kalman_gain)
