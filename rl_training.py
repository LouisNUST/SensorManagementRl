from rl_simulator import OTPSimulator
from rl_environment import OTPEnvironment
from rl_featurizers import RBFFeaturizer
from rl_sensors import ParameterizedPolicyOTPSensor, PolicyGradientParameterUpdater
from rl_targets import ConstantVelocityTarget

if __name__ == "__main__":

    featurizer = RBFFeaturizer(num_rbf_components=20, rbf_variance=1)

    environment = OTPEnvironment(bearing_variance=1E-2)

    simulator = OTPSimulator(max_num_episodes=50000, episode_length=2000)

    agent = ParameterizedPolicyOTPSensor(num_features=20, parameter_updater=PolicyGradientParameterUpdater(learning_rate=.001))

    simulator.simulate(environment, agent, featurizer, target_factory=lambda: ConstantVelocityTarget())
