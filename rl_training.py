from rl_simulator import OTPSimulator
from rl_environment import OTPEnvironment
from rl_featurizers import RBFFeaturizer
from rl_sensors import *
from rl_targets import ConstantVelocityTarget
from rl_metrics import SimulationMetrics
from rl_optimization import PolicyGradientParameterUpdater

from tempfile import TemporaryFile

import numpy as np
import os

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

    #load initial weights
    best_folder_name = "/Users/u6042446/Desktop/SensorManagementRl/results/96189/"
    initial_weights = np.load(best_folder_name+"initial_params.npz")
    w1 = initial_weights['arr_0']
    w2 = initial_weights['arr_1']
    b1 = initial_weights['arr_2']
    b2 = initial_weights['arr_3']

    #final_weights = np.load(best_folder_name + "final_params.npz")
    #w1_final = final_weights['arr_0']
    #w2_final = final_weights['arr_1']
    #b1_final = final_weights['arr_2']
    #b2_final = final_weights['arr_3']

    num_features = 20
    rbf_variance = 1
    sensor_variance = 1
    learning_rate = .001
    seed = np.random.randint(0,1E5)
    seed = -1
    if os.path.isdir("/Users/u6042446/Desktop/SensorManagementRl/results/"+str(seed)):
        pass
    else:
        os.mkdir("/Users/u6042446/Desktop/SensorManagementRl/results/"+str(seed))
    #create a folder for this seed

    #generate a random-seed

    # featurizer = RBFFeaturizer(num_rbf_components=num_features, rbf_variance=rbf_variance)
    # agent = StochasticPolicyOTPSensor(num_features=num_features,
    #                                   parameter_updater=PolicyGradientParameterUpdater(learning_rate=learning_rate),
    #                                   sigma=sensor_variance)

    # featurizer = None
    # agent = TFNeuralNetDeterministicPolicyOTPSensor(num_input=8, learning_rate=learning_rate)

    featurizer = None
    agent = TFNeuralNetStochasticPolicyOTPSensor(num_input=7, init_learning_rate=1e-3, min_learning_rate=1e-10,
                                                 learning_rate_N_max=5000, sigma=1, shuffle=True, batch_size=64,
                                                 init_pos=None, non_linearity=tf.nn.tanh, clip_norm=5.0,initial_weights=initial_weights,seed=seed)

    w1,w2,b1,b2 = agent.get_weights()
    np.savez("/Users/u6042446/Desktop/SensorManagementRl/results/"+str(seed)+"/initial_params",w1,w2,b1,b2)


    # featurizer = RBFFeaturizer(num_rbf_components=num_features, rbf_variance=rbf_variance)
    # agent = TFStochasticPolicyOTPSensor(num_input=num_features, init_learning_rate=0.001)

    # featurizer = RBFFeaturizer(num_rbf_components=num_features, rbf_variance=rbf_variance)
    # agent = TFStochasticPolicyWithSigmaOTPSensor(num_input=num_features, init_learning_rate=0.0005)

    # featurizer = None
    # agent = TFRecurrentStochasticPolicyOTPSensor(num_input=8, learning_rate=learning_rate, sigma=sensor_variance, n_hidden=50)

    # featurizer = None
    # agent = TFNeuralNetStochasticPolicyStackingOTPSensor(state_dim=3, num_states=5, init_learning_rate=1e-8,
    #                                                      min_learning_rate=1e-12, learning_rate_N_max=10000,
    #                                                      shuffle=True, batch_size=1)

    environment = OTPEnvironment(bearing_variance=1E-2)

    simulator = OTPSimulator(max_num_episodes=1000, episode_length=2000)

    simulation_metrics = SimulationMetrics(base_path="/Users/u6042446/Desktop/SensorManagementRl/results/"+str(seed)+"/",
                                           filename=str(agent) + '.txt')

    simulator.simulate(environment, agent, featurizer, simulation_metrics=simulation_metrics,
                       target_factory=lambda: ConstantVelocityTarget(.01,.01,init_pos=None, init_vel=[5, 5]))

    simulation_metrics.close_files()
    w1, w2, b1, b2 = agent.get_weights()
    np.savez("/Users/u6042446/Desktop/SensorManagementRl/results/" + str(seed) + "/final_params",w1,w2,b1,b2)
