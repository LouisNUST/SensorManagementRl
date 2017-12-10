import gym
import numpy as np
from collections import deque

from mountain_car import *

env_name = 'MountainCarContinuous-v0'
env = gym.make(env_name)

state_dim = env.observation_space.shape[0]
num_actions = env.action_space

print(state_dim)
print(num_actions)


MAX_EPISODES = 10000
MAX_STEPS = 1000

# agent = TFNeuralNetStochasticPolicyAgent(env, num_input=2, learning_rate=1e-6, sigma=1.)
agent = TFNeuralNetStochasticPolicyEpsilonGreedyAgent(env, num_input=2, learning_rate=1e-6, sigma=1.,
                                                      init_exploration=0.5, final_exploration=0.0,
                                                      anneal_steps=MAX_EPISODES)
# agent = TFRandomFeaturesStochasticPolicyAgent(env, learning_rate=0.001, sigma=1.)
# agent = TFRandomFeaturesStochasticPolicyEpsilonGreedyAgent(env, learning_rate=0.001, sigma=1.,
#                                                            init_exploration=0.5, final_exploration=0.0,
#                                                            anneal_steps=MAX_EPISODES)

episode_history = deque(maxlen=100)
for episode_counter in range(MAX_EPISODES):
    # initialize
    state = env.reset()
    total_rewards = 0

    steps = 0
    print("Episode {}".format(episode_counter+1))

    done = False
    for step_counter in range(MAX_STEPS):
        # env.render()
        action = agent.sample_action(state)
        next_state, reward, done, _ = env.step(action)

        total_rewards += reward
        # reward = -10 if done else 0.1 # normalize reward
        agent.store_rollout(state, action, reward)

        state = next_state
        steps += 1
        if done:
            break

    agent.update_model(episode_counter)

    episode_history.append(total_rewards)
    mean_rewards = np.mean(episode_history)

    if episode_counter % 100 == 0 and episode_counter > 0:
        print("%s,%s" % (episode_counter, mean_rewards))

    print("Finished after {} timesteps".format(steps))
    print("Reward for this episode: {:.2f}".format(total_rewards))
    print("Average reward for last 100 episodes: {:.2f}".format(mean_rewards))
    # if mean_rewards >= 195.0 and len(episode_history) >= 100:
    #     print("Environment {} solved after {} episodes".format(env_name, i_episode+1))
    #     break
