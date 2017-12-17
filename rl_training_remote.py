from rl_environment import OTPEnvironment
from rl_metrics import SimulationMetrics
from rl_sensors import *
from rl_simulator import OTPSimulator
from rl_targets import ConstantVelocityTarget

if __name__ == "__main__":

    featurizer = None
    agent = TFNeuralNetStochasticPolicyStackingOTPSensor(state_dim=5, num_states=5, init_learning_rate=1e-8,
                                                         min_learning_rate=1e-12, learning_rate_N_max=10000,
                                                         shuffle=True, batch_size=1)

    environment = OTPEnvironment(bearing_variance=1E-2)

    simulator = OTPSimulator(max_num_episodes=3000, episode_length=2000)

    simulation_metrics = SimulationMetrics(base_path="/output/", filename=str(agent) + '.txt')

    simulator.simulate(environment, agent, featurizer, simulation_metrics=simulation_metrics,
                       target_factory=lambda: ConstantVelocityTarget())

    simulation_metrics.close_files()
