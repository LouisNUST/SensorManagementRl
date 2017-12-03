import errno

import matplotlib.pyplot as plt
import numpy as np
import os


def mkdirs(file_name):
    if not os.path.exists(os.path.dirname(file_name)):
        try:
            os.makedirs(os.path.dirname(file_name))
        except OSError as exc:
            if exc.errno != errno.EEXIST:
                raise


def open_file_for_writing(file_name, mode='w'):
    mkdirs(file_name)
    return open(file_name, mode)


class SimulationMetrics:
    def __init__(self, base_path):
        self._base_path = base_path
        self.weight_saver1 = []
        self.weight_saver2 = []
        self.avg_reward = []
        self.var_reward = []
        self.iteration = []

    def plot(self):
        plt.plot(self.iteration, self.avg_reward, 'ko-', linewidth=3)
        plt.xlabel("iteration", size=15)
        plt.ylabel("avg. reward", size=15)
        plt.grid(True)
        plt.show()

    def save_weights(self, weights):
        self.weight_saver1.append(weights[0][0])
        self.weight_saver2.append(weights[0][1])

    def save_rewards(self, iteration, rewards):
        self.iteration.append(iteration)
        self.avg_reward.append(np.mean(rewards))
        self.var_reward.append(np.var(rewards))

    def write_metrics_to_files(self, num_features, rbf_variance, sigma, learning_rate, weights):
        post_fix = "_" + str(num_features) + "_" + str(rbf_variance) + "_" + str(sigma) + "_" + str(learning_rate)+".txt"
        writer_avg_reward = open_file_for_writing(self._base_path + "avg_reward" + post_fix)
        writer_var_reward = open_file_for_writing(self._base_path + "var_reward" + post_fix)
        writer_weight_update = open_file_for_writing(self._base_path + "weight_update" + post_fix)
        writer_final_weight = open_file_for_writing(self._base_path + "final_weight" + post_fix)

        for r in self.avg_reward:
            writer_avg_reward.write(str(r) + "\n")
        for w in self.weight_saver1:
            writer_weight_update.write(str(w) + "\n")
        w1 = list(weights[0])
        w2 = list(weights[1])
        ww1 = []
        ww2 = []
        [ww1.append(str(x)) for x in w1]
        [ww2.append(str(x)) for x in w2]
        writer_final_weight.write("\t".join(ww1) + "\n")
        writer_final_weight.write("\t".join(ww2))
        for v in self.var_reward:
            writer_var_reward.write(str(v) + "\n")
        writer_var_reward.close()
        writer_final_weight.close()
        writer_weight_update.close()
        writer_avg_reward.close()


class EpisodeMetrics:
    def __init__(self):
        self.x_est = []
        self.y_est = []
        self.x_vel_est = []
        self.y_vel_est = []
        self.x_truth = []
        self.y_truth = []
        self.x_vel_truth = []
        self.y_vel_truth = []
        self.vel_error = []
        self.pos_error = []
        self.iteration = []
        self.innovation = []

    def save(self, tracker, target):
        estimate = tracker.get_target_state_estimate()
        truth = target.get_current_location()
        self.x_est.append(estimate[0])
        self.y_est.append(estimate[1])
        self.x_vel_est.append(estimate[2])
        self.y_vel_est.append(estimate[3])
        self.x_truth.append(truth[0])
        self.y_truth.append(truth[1])
        self.x_vel_truth.append(target.get_current_velocity()[0])
        self.y_vel_truth.append(target.get_current_velocity()[1])
        self.vel_error.append(np.linalg.norm(estimate[2:4] - np.array([target.get_current_velocity()[0], target.get_current_velocity()[1]]).reshape(2, 1)))
        self.pos_error.append(np.linalg.norm(estimate[0:2] - np.array(truth).reshape(2, 1)))
        normalized_innovation = (tracker.get_innovation_list()[-1]) / tracker.get_innovation_var()[-1]
        self.innovation.append(normalized_innovation[0])
