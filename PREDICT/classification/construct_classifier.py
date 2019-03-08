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
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Lasso
import scipy
import PREDICT.addexceptions as ae
from PREDICT.classification.estimators import RankedSVM
from PREDICT.processing.AdvancedSampler import log_uniform


def construct_classifier(config, image_features):
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

    # NOTE: Function is not working anymore for regression: need
    # to move param grid creation to the create_param_grid function
    max_iter = config['max_iter']
    if 'SVM' in config['classifiers']:
        # Support Vector Machine
        classifier = construct_SVM(config, image_features)

    elif config['classifiers'] == 'SVR':
        # Support Vector Regression
        classifier = construct_SVM(config, image_features, True)

    elif config['classifiers'] == 'RF':
        # Random forest kernel
        classifier = RandomForestClassifier(verbose=0,
                                            class_weight='balanced',
                                            n_estimators=config['RFn_estimators'],
                                            min_samples_split=config['RFmin_samples_split'],
                                            max_depth=config['RFmax_depth'])

    elif config['classifiers'] == 'RFR':
        # Random forest kernel regression
        classifier = RandomForestRegressor(verbose=0,
                                           n_estimators=config['RFn_estimators'],
                                           min_samples_split=config['RFmin_samples_split'],
                                           max_depth=config['RFmax_depth'])

    elif config['classifiers'] == 'ElasticNet':
        # Elastic Net Regression
        param_grid = {'alpha': scipy.stats.uniform(loc=1.0, scale=0.5),
                      'l1_ratio': scipy.stats.uniform(loc=0.5, scale=0.4)
                      }
        classifier = ElasticNet(max_iter=max_iter)

    elif config['classifiers'] == 'Lasso':
        # LASSO Regression
        param_grid = {'alpha': scipy.stats.uniform(loc=1.0, scale=0.5)}
        classifier = Lasso(max_iter=max_iter)

    elif config['classifiers'] == 'SGD':
        # Stochastic Gradient Descent classifier
        classifier = SGDClassifier(n_iter=config['max_iter'])
        param_grid = {'loss': ['hinge', 'squared_hinge', 'modified_huber'],
                      'penalty': ['none', 'l2', 'l1']}

    elif config['classifiers'] == 'SGDR':
        # Stochastic Gradient Descent regressor
        classifier = SGDRegressor(n_iter=config['max_iter'])
        param_grid = {'alpha': scipy.stats.uniform(loc=1.0, scale=0.5),
                      'l1_ratio': scipy.stats.uniform(loc=0.5, scale=0.4),
                      'loss': ['hinge', 'squared_hinge', 'modified_huber'],
                      'penalty': ['none', 'l2', 'l1']}

    elif config['classifiers'] == 'LR':
        # Logistic Regression
        classifier = LogisticRegression(max_iter=max_iter,
                                        penalty=config['LRpenalty'],
                                        C=config['LRC'])

    return classifier


def construct_SVM(config, image_features, regression=False):
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

    max_iter = config['max_iter']
    if not regression:
        clf = SVC(class_weight='balanced', probability=True, max_iter=max_iter)
    else:
        clf = SVMR(max_iter=max_iter)

    clf.kernel = config['SVMKernel']
    clf.C = config['SVMC']
    clf.degree = config['SVMdegree']
    clf.coef0 = config['SVMcoef0']
    clf.gamma = config['SVMgamma']

    # Check if we need to use a ranked SVM
    if config['classifiers'] == 'RankedSVM':
        clf = RankedSVM()
        param_grid = {'svm': ['Poly'],
                      'degree': [2, 3, 4, 5],
                      'gamma':  scipy.stats.uniform(loc=0, scale=1e-3),
                      'coefficient': scipy.stats.uniform(loc=0, scale=1e-2),
                      }

    return clf, param_grid


def create_param_grid(config):
    ''' Create a parameter grid for the PREDICT classifiers based on the
        provided configuration. '''

    # We only need parameters from the Classification part of the config
    config = config['Classification']

    # Create grid and put in name of classifiers and maximum iterations
    param_grid = dict()
    param_grid['classifiers'] = config['classifiers']
    param_grid['max_iter'] = config['max_iter']

    # SVM parameters
    param_grid['SVMKernel'] = config['SVMKernel']
    param_grid['SVMC'] = log_uniform(loc=config['SVMC'][0],
                                     scale=config['SVMC'][1])
    param_grid['SVMdegree'] = scipy.stats.uniform(loc=config['SVMdegree'][0],
                                                  scale=config['SVMdegree'][1])
    param_grid['SVMcoef0'] = scipy.stats.uniform(loc=config['SVMcoef0'][0],
                                                 scale=config['SVMcoef0'][1])
    param_grid['SVMgamma'] = log_uniform(loc=config['SVMgamma'][0],
                                         scale=config['SVMgamma'][1])

    # RF parameters
    param_grid['RFn_estimators'] =\
        scipy.stats.randint(loc=config['RFn_estimators'][0],
                            scale=config['RFn_estimators'][1])
    param_grid['RFmin_samples_split'] =\
        scipy.stats.randint(loc=config['RFmin_samples_split'][0],
                            scale=config['RFmin_samples_split'][1])
    param_grid['RFmax_depth'] =\
        scipy.stats.randint(loc=config['RFmax_depth'][0],
                            scale=config['RFmax_depth'][1])

    # Logistic Regression parameters
    param_grid['LRpenalty'] = param_grid['LRpenalty']
    param_grid['LRC'] = log_uniform(loc=config['LRC'][0],
                                    scale=config['LRC'][1])

    return param_grid
