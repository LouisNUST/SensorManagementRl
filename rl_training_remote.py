from rl_simulator_environment import OTPSimulatorEnvironment
from rl_metrics import SimulationMetrics
from rl_sensors import *
from rl_simulator import OTPSimulator
from rl_targets import ConstantVelocityTarget

if __name__ == "__main__":

    featurizer = None
    agent = TFNeuralNetStochasticPolicyOTPSensor(num_input=8, init_learning_rate=1e-6, min_learning_rate=1e-10,
                                                 learning_rate_N_max=10000, shuffle=True, batch_size=1)

    environment = OTPSimulatorEnvironment(bearing_variance=1E-2)

    simulator = OTPSimulator(max_num_episodes=3000, episode_length=2000)

    simulation_metrics = SimulationMetrics(base_path="/output/", filename=str(agent) + '.txt')

    simulator.simulate(environment, agent, featurizer, simulation_metrics=simulation_metrics,
                       target_factory=lambda: ConstantVelocityTarget())

    simulation_metrics.close_files()
