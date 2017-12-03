import numpy as np


class PolicyGradientParameterUpdater:
    def __init__(self, learning_rate):
        self._learning_rate = learning_rate

    def update_parameters(self, weights, iteration, episode_actions, episode_states, discounted_return, num_states, sigma):
        normalized_discounted_return = discounted_return
        rate = self._gen_learning_rate(iteration=iteration, l_max=self._learning_rate, l_min=1E-8, N_max=10000)
        prev_weight = np.array(weights)
        condition = True
        #loop over all stored state/actions (trajectories)
        for e in range(0, len(episode_actions)):
            #calculate gradient
            state = np.array(episode_states[e]).reshape(num_states, 1)
            gradient = ((episode_actions[e].reshape(2, 1) - weights.dot(state)).dot(state.transpose())) / sigma**2
            if np.isnan(gradient[0][0]) or np.linalg.norm(weights[0, :]) > 1E3:
                condition = False
                break
            # an unbiased sample of return (We replace Q(s,a) by an estimate)
            adjustment_term = gradient * normalized_discounted_return[e]
            weights += rate * adjustment_term

        if not condition:
            weights = prev_weight

        return condition, weights

    def _gen_learning_rate(self, iteration, l_max, l_min, N_max):
        if iteration > N_max:
            return l_min
        alpha = 2 * l_max
        beta = np.log((alpha / l_min - 1)) / N_max
        return alpha / (1 + np.exp(beta * iteration))
