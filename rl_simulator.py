import numpy as np
from rl_trackers import EKFTracker
from rl_metrics import EpisodeMetrics


class OTPSimulator:
    def __init__(self, max_num_episodes, episode_length):
        self._max_num_episodes = max_num_episodes
        self._episode_length = episode_length

    def simulate(self, environment, agent, featurizer, simulation_metrics, target_factory):
        simulation = _OTPSimulation(window_size=50, window_lag=10)
        episode_counter = 0
        while episode_counter < self._max_num_episodes:
            episode = _OTPSimulationEpisode(gamma=.99)
            episode_metrics = EpisodeMetrics()
            target = target_factory()
            A, B = target.move()
            tracker = EKFTracker(target.get_x(), target.get_y(), 1E9, A, B, target.get_x_variance(), target.get_y_variance(), environment.bearing_variance())
            agent.reset_location()
            episode_step_counter = 0
            while episode.is_valid:
                target.update_location()
                environment.generate_bearing(target.get_current_location(), agent.get_current_location())
                tracker.update_states(agent.get_current_location(), environment.get_last_bearing_measurement())
                current_state = self._create_current_state(tracker, agent, environment.get_last_bearing_measurement())
                current_state = self._normalize_state(current_state, environment)
                if featurizer is not None:
                    current_state = featurizer.transform(current_state)
                for x in current_state:
                    if x > 1 or x < -1:
                        episode.is_valid = False
                # update the location of sensor based on the current state
                agent.update_location(np.array(current_state))
                episode.states.append(current_state)
                # episode.update_reward_by_location_mse(simulation, target, tracker)
                episode.update_reward_by_uncertainty(simulation, tracker)
                episode.update_discounted_return()
                episode_metrics.save(episode_step_counter, tracker, target, agent)
                if episode_step_counter > self._episode_length:
                    break
                episode_step_counter += 1

            if episode.is_valid:
                condition = agent.update_parameters(episode_counter, episode.discounted_return, episode.states)

                if condition:
                    simulation.rewards.append(sum(episode.reward))
                    print("%s,%s" % (episode_counter, np.mean(simulation.rewards)))
                    simulation.sigmas.append(np.mean(agent.get_sigmas(), axis=0))
                    simulation_metrics.save_raw_reward(episode_counter, sum(episode.reward))
                    simulation_metrics.save_locations(episode_counter, episode_metrics)
                    if episode_counter % 100 == 0 and episode_counter > 0:
                        # print("%s,%s" % (episode_counter, np.mean(simulation.rewards)))
                        simulation_metrics.save_rewards(episode_counter, simulation.rewards)
                        simulation_metrics.save_sigmas(episode_counter, simulation.sigmas)
                        simulation.rewards = []
                        simulation.sigmas = []
                    episode_counter += 1
                    # simulation_metrics.save_weights(agent.get_weights())

    def _create_current_state(self, tracker, agent, last_bearing_measurement):
        # create current state s(t): target_state + sensor_state + bearing measurement + range
        target_state = list(tracker.get_target_state_estimate().reshape(len(tracker.get_target_state_estimate())))
        agent_state = list(agent.get_current_location())
        agent_to_target_estimate_range = [np.linalg.norm(agent.get_current_location() - tracker.get_target_state_estimate()[0:2])]
        return target_state + agent_state + [last_bearing_measurement] + agent_to_target_estimate_range

    def _normalize_state(self, state, environment):
        max_node = np.array([environment.get_x_max(), environment.get_y_max()])
        min_node = np.array([environment.get_x_min(), environment.get_y_min()])
        max_distance = np.linalg.norm(max_node - min_node)
        x_slope = 2.0 / (environment.get_x_max() - environment.get_x_min())
        y_slope = 2.0 / (environment.get_y_max() - environment.get_y_min())
        vel_slope = 2.0 / (environment.get_vel_max() - environment.get_vel_min())
        measure_slope = 1.0 / np.pi
        distance_slope = 2.0 / max_distance
        # normalization (map each value to the bound (-1,1)
        state[0] = -1 + x_slope * (state[0] - environment.get_x_min())
        state[1] = -1 + y_slope * (state[1] - environment.get_y_min())
        state[2] = -1 + vel_slope * (state[2] - environment.get_vel_min())
        state[3] = -1 + vel_slope * (state[3] - environment.get_vel_min())
        state[4] = -1 + x_slope * (state[4] - environment.get_x_min())
        state[5] = -1 + y_slope * (state[5] - environment.get_y_min())
        state[6] = -1 + measure_slope * state[6]
        state[7] = -1 + distance_slope * state[7]
        return state


class _OTPSimulation:
    def __init__(self, window_size, window_lag):
        self.window_size = window_size
        self.window_lag = window_lag
        self.rewards = []
        self.sigmas = []


class _OTPSimulationEpisode:
    def __init__(self, gamma):
        self._gamma = gamma  # discount factor
        self.is_valid = True
        self.discounted_return = np.array([])
        self.discount_vector = np.array([])
        self.reward = []
        self.uncertainty = []
        self.states = []
        self.location_mse = []

    def update_reward_by_location_mse(self, simulation, target, tracker):
        true_target_location = target.get_current_location()
        target_location_estimate = tracker.get_target_state_estimate()[0:2].reshape(2)
        mse = self.mse(target_location_estimate, true_target_location)
        self.location_mse.append(mse)
        if len(self.location_mse) < simulation.window_size + simulation.window_lag:
            self.reward.append(0)
        else:
            current_avg = np.mean(self.location_mse[-simulation.window_size:])
            prev_avg = np.mean(self.location_mse[-(simulation.window_size + simulation.window_lag):-simulation.window_lag])
            if current_avg < prev_avg:
                self.reward.append(1)
            else:
                self.reward.append(0)

    def update_reward_by_uncertainty(self, simulation, tracker):
        unnormalized_uncertainty = np.sum(tracker.get_estimation_error_covariance_matrix().diagonal())
        # reward: see if the uncertainty has decayed or if it has gone below a certain value
        self.uncertainty.append((1.0/tracker.get_max_uncertainty()) * unnormalized_uncertainty)
        if len(self.uncertainty) < simulation.window_size + simulation.window_lag:
            self.reward.append(0)
        else:
            current_avg = np.mean(self.uncertainty[-simulation.window_size:])
            prev_avg = np.mean(self.uncertainty[-(simulation.window_size + simulation.window_lag):-simulation.window_lag])
            if current_avg < prev_avg or self.uncertainty[-1] < .1:
                self.reward.append(1)
            else:
                self.reward.append(0)

    def update_discounted_return(self):
        self.discount_vector = self._gamma * np.array(self.discount_vector)
        self.discounted_return += (1.0 * self.reward[-1]) * self.discount_vector
        new_return = 1.0 * self.reward[-1]
        list_discounted_return = list(self.discounted_return)
        list_discounted_return.append(new_return)
        self.discounted_return = np.array(list_discounted_return)
        list_discount_vector = list(self.discount_vector)
        list_discount_vector.append(1)
        self.discount_vector = np.array(list_discount_vector)

    def mse(self, x, y):
        return ((x - y) ** 2).mean()
