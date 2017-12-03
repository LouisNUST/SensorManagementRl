import sklearn
from sklearn.kernel_approximation import RBFSampler
import numpy as np


class RBFFeaturizer:
    def __init__(self, num_rbf_components, rbf_variance):
        # Read all randomly-generated samples for feature-generation (this is done for matching and feature construction)
        list_of_states = []
        with open("sampled_states","r") as f:
            for line in f:
                data = line.strip().split("\t")
                dd = []
                [dd.append(float(x)) for x in data]
                list_of_states.append(dd)

        self._featurizer = sklearn.pipeline.FeatureUnion([("rbf1", RBFSampler(gamma=rbf_variance, n_components=num_rbf_components, random_state=1))])
        self._featurizer.fit(np.array(list_of_states))  # Use this featurizer for normalization
