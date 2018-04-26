#!/usr/bin/env python

# Copyright 2017-2018 Biomedical Imaging Group Rotterdam, Departments of
# Medical Informatics and Radiology, Erasmus MC, Rotterdam, The Netherlands
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from sklearn.svm import SVC
from sklearn.svm import SVR as SVMR
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import SGDClassifier, ElasticNet, SGDRegressor
from sklearn.linear_model import Lasso
import scipy


def construct_classifier(config):
    """Interface to create classification

    Different classifications can be created using this common interface

    Parameters
    ----------
        config: dict, mandatory
                Contains the required config settings. See the Github Wiki for
                all available fields.

    Returns:
        Constructed classifier
    """

    if config['Classification']['classifier'] == 'SVM':
        # Support Vector Machine
        classifier, param_grid = construct_SVM(config)

    elif config['Classification']['classifier'] == 'SVR':
        # Support Vector Regression
        classifier, param_grid = construct_SVM(config, True)

    elif config['Classification']['classifier'] == 'RF':
        # Random forest kernel
        param_grid = {'n_estimators': scipy.stats.randint(low=300, high=500),
                      'min_samples_split': scipy.stats.randint(low=2, high=5),
                      'max_depth': scipy.stats.randint(low=5, high=10)}
        classifier = RandomForestClassifier(verbose=0,
                                            class_weight='balanced')

    elif config['Classification']['classifier'] == 'RFR':
        # Random forest kernel regression
        param_grid = {'n_estimators': scipy.stats.randint(low=300, high=500),
                      'min_samples_split': scipy.stats.randint(low=2, high=5),
                      'max_depth': scipy.stats.randint(low=5, high=10)}
        classifier = RandomForestRegressor(verbose=0)

    elif config['Classification']['classifier'] == 'ElasticNet':
        # Elastic Net Regression
        param_grid = {'alpha': scipy.stats.uniform(loc=1.0, scale=0.5),
                      'l1_ratio': scipy.stats.uniform(loc=0.5, scale=0.4)
                      }
        classifier = ElasticNet(max_iter=100000)

    elif config['Classification']['classifier'] == 'Lasso':
        # LASSO Regression
        param_grid = {'alpha': scipy.stats.uniform(loc=1.0, scale=0.5)}
        classifier = Lasso(max_iter=100000)

    elif config['Classification']['classifier'] == 'SGD':
        # Stochastic Gradient Descent classifier
        classifier = SGDClassifier(n_iter=config['Classification']['n_epoch'])
        param_grid = {'loss': ['hinge', 'squared_hinge', 'modified_huber'],
                      'penalty': ['none', 'l2', 'l1']}

    elif config['Classification']['classifier'] == 'SGDR':
        # Stochastic Gradient Descent regressor
        classifier = SGDRegressor(n_iter=config['Classification']['n_epoch'])
        param_grid = {'alpha': scipy.stats.uniform(loc=1.0, scale=0.5),
                      'l1_ratio': scipy.stats.uniform(loc=0.5, scale=0.4),
                      'loss': ['hinge', 'squared_hinge', 'modified_huber'],
                      'penalty': ['none', 'l2', 'l1']}

    return classifier, param_grid


def construct_SVM(config, regression=False):
    """
    Constructs a SVM classifier

    Args:
        config (dict): Dictionary of the required config settings
        mutation_data (dict): Mutation data that should be classified
        features (pandas dataframe): A pandas dataframe containing the features
         to be used for classification

    Returns:
        SVM/SVR classifier, parameter grid
    """

    # TODO: move the max_iter parameter to main config
    if not regression:
        clf = SVC(class_weight='balanced', probability=True, max_iter=100000)
    else:
        clf = SVMR(max_iter=100000)

    if config['Classification']['Kernel'] == "polynomial" or config['Classification']['Kernel'] == "poly":
        param_grid = {'kernel': ['poly'], 'C': scipy.stats.uniform(loc=0.5e8, scale=0.5e8), 'degree': scipy.stats.uniform(loc=3, scale=2), 'coef0': scipy.stats.uniform(loc=0.5, scale=0.5)}

    elif config['Classification']['Kernel'] == "linear":
        param_grid = {'kernel': ['linear'], 'C': scipy.stats.uniform(loc=0.5e8, scale=0.5e8), 'coef0': scipy.stats.uniform(loc=0.5, scale=0.5)}

    elif config['Classification']['Kernel'] == "rbf":
        param_grid = {'kernel': ['rbf'], 'gamma':  scipy.stats.uniform(loc=0.5e-3, scale=0.5e-3), 'nu': scipy.stats.uniform(loc=0.5, scale=0.5)}
    else:
        raise KeyError("{} is not a valid SVM kernel type!").format(config['Classification']['Kernel'])

    return clf, param_grid
