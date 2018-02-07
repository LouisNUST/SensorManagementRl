import numpy as np
import sklearn
import sklearn.pipeline
from sklearn.kernel_approximation import RBFSampler
from scipy.stats import norm


class RBFFeaturizer:
    def __init__(self, num_rbf_components, rbf_variance):
        """

        :param num_rbf_components: number of rbf-components
        :param rbf_variance:
        """
        # Read all randomly-generated samples for feature-generation (this is done for matching and feature construction)
        list_of_states = []
        with open("sampled_states", "r") as f:
            for line in f:
                data = line.strip().split("\t")
                dd = []
                [dd.append(float(x)) for x in data]
                list_of_states.append(dd)

        self._featurizer = sklearn.pipeline.FeatureUnion([("rbf1", RBFSampler(gamma=rbf_variance, n_components=num_rbf_components, random_state=1))])
        self._featurizer.fit(np.array(list_of_states))  # Use this featurizer for normalization

    def transform(self, current_state):
        transformed = self._featurizer.transform(np.array(current_state).reshape(1, len(current_state)))
        return list(transformed[0])

class singleRBFTile:
    def __init__(self,num_sub_grids,num_tiles):
        """

        :param num_sub_grids: number of grids inside each box
        :param num_tiles: Number of boxes (RBF boxes)
        """
        self.state_dim = num_tiles
        self.num_sub_grids = num_sub_grids


    def transform(self,min_val,max_val,vals):
        """
        :param min_val: minimum-value
        :param max_val: maximum-value
        :param vals: vector of values
        :return:
        """

        #normalization
        step_size = (max_val-min_val)/(1.0*self.num_sub_grids*self.state_dim)#based on the normalized value
        #locate means
        rbf_grid_width = self.num_sub_grids

        #populate means and standard-deviations of RBFs
        means = []
        stds = []
        for n in range(0,self.state_dim):
            current_mean = 0+(n+.5)*rbf_grid_width
            means.append(current_mean)
            stds.append(rbf_grid_width/2.0)



        #calculate memberships

        features = []
        for v in vals:
            grid_index = (v-min_val)/step_size
            memberships = []
            [memberships.append(round(norm.pdf(grid_index,means[idx],stds[idx]),4)) for idx in range(0,self.state_dim)]
            features.append(memberships)

        features = np.array(features)
        features = np.sum(features,axis=0)

        if (np.linalg.norm(features)!=0):
            normalized_feature = features / np.linalg.norm(features)
        else:
            normalized_feature = features
        return (normalized_feature)


if __name__=="__main__":
    object = singleRBFTile(1000,20)
    f = object.transform(-10,10,[2.345,1.17])

