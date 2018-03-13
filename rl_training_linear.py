from rl_simulator import OTPSimulator
from rl_simulator_environment import OTPSimulatorEnvironment
from rl_sensors import *
from rl_targets import ConstantVelocityTarget
from rl_metrics import SimulationMetrics
from rl_rewards import *
from rl_utils import *

import tensorflow as tf

if __name__ == "__main__":

    num_input = 6
    init_learning_rate = 1e-3
    min_learning_rate = 1e-8
    learning_rate_N_max = 10000
    sensor_sigma = 1.
    optimizer = tf.train.GradientDescentOptimizer
    featurizer = None

    sensor_init_pos = None
    target_init_pos = None
    target_init_vel = None
    target_x_variance = 0
    target_y_variance = 0

    bearing_variance = 1e-2

    max_num_episodes = 10000
    episode_length = 1500
    use_true_target_state = True

    gamma = .99
    reward_strategy = RewardByDistanceDiscrete()
    reward_printer = PeriodicAverageRewardPrinter(window=100)

    agent = TFLinearStochasticPolicyOTPSensor(num_input=num_input, init_learning_rate=init_learning_rate,
                                              min_learning_rate=min_learning_rate,
                                              learning_rate_N_max=learning_rate_N_max, init_sigma=sensor_sigma,
                                              init_pos=sensor_init_pos, optimizer=optimizer)

    environment = OTPSimulatorEnvironment(bearing_variance=bearing_variance)

    simulator = OTPSimulator(max_num_episodes=max_num_episodes, episode_length=episode_length, state_size=num_input,
                             use_true_target_state=use_true_target_state)

    simulation_metrics = SimulationMetrics(base_path="/Users/u6046782/SensorManagementRl/out/",
                                           filename=str(agent) + '.txt')

    simulator.simulate(environment, agent, featurizer, simulation_metrics=simulation_metrics,
                       target_factory=lambda: ConstantVelocityTarget(init_pos=target_init_pos, init_vel=target_init_vel,
                                                                     x_variance=target_x_variance, y_variance=target_y_variance),
                       reward_strategy=reward_strategy, gamma=gamma, reward_printer=reward_printer)

    simulation_metrics.close_files()
