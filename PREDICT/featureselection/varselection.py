from sklearn.base import BaseEstimator
from sklearn.feature_selection.base import SelectorMixin
import numpy as np


class varselection(BaseEstimator, SelectorMixin):

    def __init__(self, threshold):
        self.threshold = threshold


    def fit(self, image_features):
        selectrows = list()
        means = np.mean(image_features, axis=0)
        variances = np.var(image_features, axis=0)

        for i in range(image_features.shape[1]):
            if variances[i] > self.threshold*(1-self.threshold)*means[i]:
                selectrows.append(i)

        self.selectrows = selectrows
        return self


    def transform(self, inputarray):
        '''
        Transform the inputarray to select only the features based on the
        result from the fit function.

        Parameters
        ----------
        inputarray: numpy array, mandatory
                Array containing the items to use selection on. The type of
                item in this list does not matter, e.g. floats, strings etc.
        '''
        return np.asarray([np.asarray(x)[self.selectrows].tolist() for x in inputarray])

    def _get_support_mask(self):
        # NOTE: Method is required for the Selector class, but can be empty
        pass
