from rl_simulator import OTPSimulator
from rl_environment import OTPEnvironment
from rl_featurizers import RBFFeaturizer
from rl_sensors import *
from rl_targets import ConstantVelocityTarget
from rl_metrics import SimulationMetrics
from rl_optimization import PolicyGradientParameterUpdater

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

    num_features = 20
    rbf_variance = 1
    sensor_variance = 1
    learning_rate = .001

    # featurizer = RBFFeaturizer(num_rbf_components=num_features, rbf_variance=rbf_variance)
    # agent = StochasticPolicyOTPSensor(num_features=num_features,
    #                                   parameter_updater=PolicyGradientParameterUpdater(learning_rate=learning_rate),
    #                                   sigma=sensor_variance)

    # featurizer = None
    # agent = TFNeuralNetDeterministicPolicyOTPSensor(num_input=8, learning_rate=learning_rate)

    featurizer = None
    agent = TFNeuralNetStochasticPolicyOTPSensor(num_input=8, init_learning_rate=1e-6, min_learning_rate=1e-10,
                                                 learning_rate_N_max=10000, shuffle=True, batch_size=1)

    # featurizer = RBFFeaturizer(num_rbf_components=num_features, rbf_variance=rbf_variance)
    # agent = TFStochasticPolicyOTPSensor(num_input=num_features, init_learning_rate=0.001)

    # featurizer = None
    # agent = TFRecurrentStochasticPolicyOTPSensor(num_input=8, learning_rate=learning_rate, sigma=sensor_variance, n_hidden=50)

    # featurizer = None
    # agent = TFNeuralNetStochasticPolicyStackingOTPSensor(num_input=8, learning_rate=1e-6, sigma=sensor_variance)

    environment = OTPEnvironment(bearing_variance=1E-2)

    simulator = OTPSimulator(max_num_episodes=50000, episode_length=2000)

    simulation_metrics = SimulationMetrics(base_path="/Users/u6046782/SensorManagementRl/out/",
                                           filename=str(agent) + '.txt')

    simulator.simulate(environment, agent, featurizer, simulation_metrics=simulation_metrics,
                       target_factory=lambda: ConstantVelocityTarget())

    simulation_metrics.close_files()
