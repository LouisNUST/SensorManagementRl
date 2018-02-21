import numpy as np

from file_helper import *


class SimulationMetrics:
    def __init__(self, base_path, filename):
        self._base_path = base_path
        avg_reward_filename = self._base_path + "avg_reward_" + filename
        var_reward_filename = self._base_path + "var_reward_" + filename
        raw_reward_filename = self._base_path + "raw_reward_" + filename
        locations_filename = self._base_path + "locations_" + filename
        sensor_sigmas_filename = self._base_path + "sensor_sigmas_" + filename
        initial_parameters_filename = self._base_path + "initial_parameters_"+filename
        final_weights_file_name = self._base_path+"final_weights_"+ filename

        silent_remove(avg_reward_filename)
        silent_remove(var_reward_filename)
        silent_remove(raw_reward_filename)
        silent_remove(locations_filename)
        silent_remove(sensor_sigmas_filename)
        silent_remove(initial_parameters_filename)
        silent_remove(final_weights_file_name)

        self._writer_avg_reward = open_file_for_writing(avg_reward_filename, mode="a")
        self._writer_var_reward = open_file_for_writing(var_reward_filename, mode="a")
        self._writer_raw_reward = open_file_for_writing(raw_reward_filename, mode="a")
        #self._writer_locations = open_file_for_writing(locations_filename, mode="a")
        #self._writer_sigmas = open_file_for_writing(sensor_sigmas_filename, mode="a")
        #self._writer_initial_params = open_file_for_writing(initial_parameters_filename, mode = "a")
        #self._writer_final_weights = open_file_for_writing(final_weights_file_name, mode="a")

    def save_raw_reward(self, episode_number, reward):
        self._writer_raw_reward.write("%s,%s\n" % (episode_number, reward))
        self._flush(self._writer_raw_reward)

    def save_rewards(self, episode_number, rewards):
        self._writer_avg_reward.write("%s,%s\n" % (episode_number, np.mean(rewards)))
        self._flush(self._writer_avg_reward)
        self._writer_var_reward.write("%s,%s\n" % (episode_number, np.var(rewards)))
        self._flush(self._writer_var_reward)

    def save_locations(self, episode_number, episode_metrics):
        for i in range(len(episode_metrics.iteration)):
            self._writer_locations.write("%s,%s,%s,%s,%s,%s,%s,%s,%s\n" % (episode_number, episode_metrics.iteration[i],
                                                                           episode_metrics.sensor_x[i], episode_metrics.sensor_y[i],
                                                                           episode_metrics.target_x_truth[i], episode_metrics.target_y_truth[i],
                                                                           episode_metrics.target_x_est[i], episode_metrics.target_y_est[i],
                                                                           episode_metrics.bearing[i]))
        self._flush(self._writer_locations)

    def save_sigmas(self, episode_number, sigmas):
        self._writer_sigmas.write("%s,%s\n" % (episode_number, np.mean(sigmas, axis=0)))
        self._flush(self._writer_sigmas)



    def _flush(self, f):
        f.flush()
        os.fsync(f)

    def close_files(self):
        self._writer_avg_reward.close()
        self._writer_var_reward.close()


class EpisodeMetrics:
    def __init__(self):
        self.target_x_est = []
        self.target_y_est = []
        self.x_vel_est = []
        self.y_vel_est = []
        self.target_x_truth = []
        self.target_y_truth = []
        self.x_vel_truth = []
        self.y_vel_truth = []
        self.vel_error = []
        self.pos_error = []
        self.iteration = []
        self.innovation = []
        self.sensor_x = []
        self.sensor_y = []
        self.bearing = []

    def save(self, iteration, tracker, target, sensor, bearing):
        estimate = tracker.get_target_state_estimate()
        truth = target.get_current_location()
        self.target_x_est.append(estimate[0][0])
        self.target_y_est.append(estimate[1][0])
        self.x_vel_est.append(estimate[2])
        self.y_vel_est.append(estimate[3])
        self.target_x_truth.append(truth[0])
        self.target_y_truth.append(truth[1])
        self.x_vel_truth.append(target.get_current_velocity()[0])
        self.y_vel_truth.append(target.get_current_velocity()[1])
        self.vel_error.append(np.linalg.norm(estimate[2:4] - np.array([target.get_current_velocity()[0], target.get_current_velocity()[1]]).reshape(2, 1)))
        self.pos_error.append(np.linalg.norm(estimate[0:2] - np.array(truth).reshape(2, 1)))
        normalized_innovation = (tracker.get_innovation_list()[-1]) / tracker.get_innovation_var()[-1]
        self.innovation.append(normalized_innovation[0])
        self.iteration.append(iteration)
        self.sensor_x.append(sensor.get_current_location()[0])
        self.sensor_y.append(sensor.get_current_location()[1])
        self.bearing.append(bearing)
