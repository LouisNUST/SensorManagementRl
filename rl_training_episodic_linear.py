import tensorflow as tf

from rl_metrics import SimulationMetrics
from rl_rewards import *
from rl_sensors import *
from rl_simulation import EpisodicSimulation
from rl_simulator_environment import OTPSimulatorEnvironment
from rl_targets import ConstantVelocityTarget
from rl_utils import *

if __name__ == "__main__":

    num_input = 6
    init_learning_rate = 1e-3
    min_learning_rate = 1e-10
    learning_rate_N_max = 3000
    sensor_sigma = 1
    batch_size = 1
    shuffle = True
    reduction = tf.reduce_mean

    optimizer = tf.train.GradientDescentOptimizer
    featurizer = None

    sensor_init_pos = [2000, 0]
    target_init_pos = [0, 0]
    target_init_vel = [5, 5]
    target_x_variance = 0
    target_y_variance = 0

    bearing_variance = 1e-2

    max_num_iterations = 150
    num_episodes_per_iteration = 20
    num_steps_per_episode = 2000
    use_true_target_state = False

    gamma = .99
    reward_strategy = RewardByDistance()
    reward_printer = PeriodicAverageRewardPrinter(window=20)

    agent = TFLinearStochasticBatchPolicyOTPSensor(num_input=num_input, init_learning_rate=init_learning_rate,
                                                   min_learning_rate=min_learning_rate,
                                                   learning_rate_N_max=learning_rate_N_max, sigma=sensor_sigma,
                                                   batch_size=batch_size, shuffle=shuffle, init_pos=sensor_init_pos,
                                                   reduction=reduction, optimizer=optimizer)

    environment = OTPSimulatorEnvironment(bearing_variance=bearing_variance)

    simulator = EpisodicSimulation(max_num_iterations=max_num_iterations, num_episodes_per_iteration=num_episodes_per_iteration,
                                   num_steps_per_episode=num_steps_per_episode, state_size=num_input,
                                   use_true_target_state=use_true_target_state)

    simulation_metrics = SimulationMetrics(base_path="/Users/u6046782/SensorManagementRl/out/",
                                           filename=str(agent) + '.txt')

    simulator.simulate(environment, agent, featurizer, simulation_metrics=simulation_metrics,
                       target_factory=lambda: ConstantVelocityTarget(init_pos=target_init_pos, init_vel=target_init_vel,
                                                                     x_variance=target_x_variance, y_variance=target_y_variance),
                       reward_strategy=reward_strategy, gamma=gamma, reward_printer=reward_printer)

    simulation_metrics.close_files()
