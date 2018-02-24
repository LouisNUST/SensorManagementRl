from rl_simulator import OTPSimulator
from rl_environment import OTPEnvironment
from rl_featurizers import RBFFeaturizer
from rl_sensors import *
from rl_targets import ConstantVelocityTarget
from rl_metrics import SimulationMetrics
from rl_optimization import PolicyGradientParameterUpdater

import tensorflow as tf

# the Target state consists of: x (x coordinate), y (y coordinate), xdot (velocity in the x dimension), ydot (velocity in the y dimension)
# the Sensor state consists of: x (x coordinate), y (y coordinate)
# additionally, we keep track of the bearing from the Sensor to the Target (noisy),
#   and the range from the Sensor to the estimated position of the Target (estimated with the EKF)
# The overall system state consists of:
#   estimated Target x,
#   estimated Target y,
#   estimated Target xdot,
#   estimated Target ydot,
#   Sensor x,
#   Sensor y,
#   bearing (noisy),
#   range (Sensor x, y to estimated Target x, y)
# Furthermore, this system state is featurized using an RBF sampler, into a vector of a pre-specified number of
#   features (namely, 20, but it could be any number), where each value in the vector is a number in [0, 1].

if __name__ == "__main__":

    num_input = 7
    init_learning_rate = 1e-3
    min_learning_rate = 1e-10
    learning_rate_N_max = 3000
    sensor_sigma = 1
    shuffle = True
    batch_size = 64
    non_linearity = tf.nn.tanh
    clip_norm = 5.0
    featurizer = None

    sensor_init_pos = [2000, 0]
    target_init_pos = [0, 0]
    target_init_vel = [5, 5]

    bearing_variance = 1e-2

    max_num_episodes = 3000
    episode_length = 2000

    agent = TFNeuralNetStochasticPolicyOTPSensor(num_input=num_input, init_learning_rate=init_learning_rate,
                                                 min_learning_rate=min_learning_rate,
                                                 learning_rate_N_max=learning_rate_N_max, sigma=sensor_sigma,
                                                 shuffle=shuffle, batch_size=batch_size, init_pos=sensor_init_pos,
                                                 non_linearity=non_linearity, clip_norm=clip_norm)

    environment = OTPEnvironment(bearing_variance=bearing_variance)

    simulator = OTPSimulator(max_num_episodes=max_num_episodes, episode_length=episode_length)

    simulation_metrics = SimulationMetrics(base_path="/Users/u6046782/SensorManagementRl/out/",
                                           filename=str(agent) + '.txt')

    simulator.simulate(environment, agent, featurizer, simulation_metrics=simulation_metrics,
                       target_factory=lambda: ConstantVelocityTarget(init_pos=target_init_pos, init_vel=target_init_vel))

    simulation_metrics.close_files()
