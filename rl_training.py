from rl_simulator import OTPSimulator
from rl_environment import OTPEnvironment
from rl_featurizers import RBFFeaturizer
from rl_sensors import ParameterizedPolicyOTPSensor
from rl_optimization import PolicyGradientParameterUpdater
from rl_targets import ConstantVelocityTarget
from rl_metrics import SimulationMetrics

if __name__ == "__main__":

    num_features = 20
    rbf_variance = 1
    sensor_variance = 1
    learning_rate = .001

    featurizer = RBFFeaturizer(num_rbf_components=num_features, rbf_variance=rbf_variance)

    environment = OTPEnvironment(bearing_variance=1E-2)

    simulator = OTPSimulator(max_num_episodes=50000, episode_length=2000)

    agent = ParameterizedPolicyOTPSensor(num_features=num_features,
                                         parameter_updater=PolicyGradientParameterUpdater(learning_rate=learning_rate),
                                         sigma=sensor_variance)

    simulation_metrics = SimulationMetrics(base_path="/Users/u6046782/SensorManagementRl/out/")

    simulator.simulate(environment, agent, featurizer, simulation_metrics,
                       target_factory=lambda: ConstantVelocityTarget())

    simulation_metrics.write_metrics_to_files(num_features, rbf_variance, sensor_variance, learning_rate, agent.get_weights())

    simulation_metrics.plot()
