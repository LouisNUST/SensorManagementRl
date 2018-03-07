import numpy as np
from rl_trackers import EKFTracker
from rl_metrics import EpisodeMetrics
from rl_rewards import RewardByTrace
from rl_utils import RawRewardPrinter


class OTPSimulator:
    def __init__(self, max_num_episodes, episode_length, state_size=7, use_true_target_state=False):
        self._max_num_episodes = max_num_episodes
        self._episode_length = episode_length
        self._use_true_target_state = use_true_target_state
        self._state_size = state_size

    def simulate(self, environment, agent, featurizer, simulation_metrics, target_factory,
                 reward_strategy=RewardByTrace(), gamma=.99, reward_printer=RawRewardPrinter()):
        simulation = _OTPSimulation()
        episode_counter = 0
        while episode_counter < self._max_num_episodes:
            episode = _OTPSimulationEpisode(gamma=gamma, reward_strategy=reward_strategy)
            episode_metrics = EpisodeMetrics()
            target = target_factory()
            A, B = target.move()
            tracker = EKFTracker(target.get_x(), target.get_y(), 1E9, A, B, target.get_x_variance(), target.get_y_variance(), environment.bearing_variance())
            agent.reset_location()
            episode_step_counter = 0
            while True:
                target.update_location()
                environment.generate_bearing(target.get_current_location(), agent.get_current_location())
                tracker.update_states(agent.get_current_location(), environment.get_last_bearing_measurement())
                current_state = self._create_current_state(tracker, agent, environment.get_last_bearing_measurement(), target)
                current_state = self._normalize_state(current_state, environment)
                if featurizer is not None:
                    current_state = featurizer.transform(current_state)
                agent.update_location(np.array(current_state))
                episode.states.append(current_state)
                episode.update_reward(agent, target, tracker)
                episode.update_discounted_return()
                episode_metrics.save(episode_step_counter, tracker, target, agent, environment.get_last_bearing_measurement())
                if episode_step_counter > self._episode_length:
                    break
                episode_step_counter += 1

            valid_update = agent.update_parameters(episode_counter, episode.discounted_return, episode.states)

            if valid_update:
                simulation.rewards.append(sum(episode.reward))
                reward_printer.print_reward(episode_counter, episode.reward)
                simulation.sigmas.append(np.mean(agent.get_sigmas(), axis=0))
                simulation_metrics.save_raw_reward(episode_counter, sum(episode.reward))
                simulation_metrics.save_locations(episode_counter, episode_metrics)
                if episode_counter % reward_printer.get_window() == 0 and episode_counter > 0:
                    simulation_metrics.save_rewards(episode_counter, simulation.rewards)
                    simulation_metrics.save_sigmas(episode_counter, simulation.sigmas)
                    simulation.rewards = []
                    simulation.sigmas = []
                episode_counter += 1
                # simulation_metrics.save_weights(agent.get_weights())

    def _create_current_state(self, tracker, agent, last_bearing_measurement, target):
        # create current state s(t): target_state + sensor_state + bearing measurement + range
        if self._use_true_target_state:
            target_state = [target.get_x(), target.get_y(), target.get_x_dot(), target.get_y_dot()]
        else:
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

        # ensure target estimate values are in allowed ranges
        state[0] = np.clip(state[0], environment.get_x_min(), environment.get_x_max())
        state[1] = np.clip(state[1], environment.get_y_min(), environment.get_y_max())
        state[2] = np.clip(state[2], environment.get_vel_min(), environment.get_vel_max())
        state[3] = np.clip(state[3], environment.get_vel_min(), environment.get_vel_max())

        if self._state_size == 7:
            new_state = [None]*7
            # normalization (map each value to the bound (-1,1)
            new_state[0] = -1 + x_slope * (state[0] - environment.get_x_min())
            new_state[1] = -1 + y_slope * (state[1] - environment.get_y_min())
            new_state[2] = -1 + vel_slope * (state[2] - environment.get_vel_min())
            new_state[3] = -1 + vel_slope * (state[3] - environment.get_vel_min())
            new_state[4] = -1 + x_slope * (state[4] - environment.get_x_min())
            new_state[5] = -1 + y_slope * (state[5] - environment.get_y_min())
            new_state[6] = np.clip(-1 + measure_slope * state[6], -1., 1.)
            # new_state[7] = -1 + distance_slope * state[7]
        elif self._state_size == 6:
            new_state = [None]*6
            # normalization (map each value to the bound (-1,1)
            new_state[0] = -1 + x_slope * (state[0] - environment.get_x_min())
            new_state[1] = -1 + y_slope * (state[1] - environment.get_y_min())
            new_state[2] = -1 + vel_slope * (state[2] - environment.get_vel_min())
            new_state[3] = -1 + vel_slope * (state[3] - environment.get_vel_min())
            new_state[4] = -1 + x_slope * (state[4] - environment.get_x_min())
            new_state[5] = -1 + y_slope * (state[5] - environment.get_y_min())
        elif self._state_size == 5:
            new_state = [None]*5
            # normalization (map each value to the bound (-1,1)
            new_state[0] = -1 + x_slope * (state[0] - environment.get_x_min())
            new_state[1] = -1 + y_slope * (state[1] - environment.get_y_min())
            new_state[2] = -1 + x_slope * (state[4] - environment.get_x_min())
            new_state[3] = -1 + y_slope * (state[5] - environment.get_y_min())
            new_state[4] = np.clip(-1 + measure_slope * state[6], -1., 1.)
        else:
            raise Exception("unsupported state size: %s" % self._state_size)

        return new_state


class _OTPSimulation:
    def __init__(self):
        self.rewards = []
        self.sigmas = []


class _OTPSimulationEpisode:
    def __init__(self, gamma, reward_strategy):
        self._gamma = gamma  # discount factor
        self.discounted_return = np.array([])
        self.discount_vector = np.array([])
        self.reward_strategy = reward_strategy
        self.reward = []
        self.states = []

    def update_reward(self, sensor, target, tracker):
        self.reward.append(self.reward_strategy.get_reward(sensor, target, tracker))

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
