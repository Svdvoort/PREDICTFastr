#!/usr/bin/env python

# Copyright 2017-2019 Biomedical Imaging Group Rotterdam, Departments of
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

import numpy as np
from sklearn.utils import check_random_state
from sklearn.model_selection import StratifiedShuffleSplit, ShuffleSplit
from PREDICT.processing.SearchCV import RandomizedSearchCVfastr, RandomizedSearchCVJoblib


def random_search_parameters(features, labels, N_iter, test_size,
                             param_grid, scoring_method,
                             n_jobspercore=200, use_fastr=False,
                             n_cores=1, fastr_plugin=None):
    """
    Train a classifier and simultaneously optimizes hyperparameters using a
    randomized search.

    Arguments:
        features: numpy array containing the training features.
        labels: list containing the object labels to be trained on.
        N_iter: integer listing the number of iterations to be used in the
                hyperparameter optimization.
        test_size: float listing the test size percentage used in the cross
                   validation.
        classifier: sklearn classifier to be tested
        param_grid: dictionary containing all possible hyperparameters and their
                    values or distrubitions.
        scoring_method: string defining scoring method used in optimization,
                        e.g. f1_weighted for a SVM.
        n_jobsperscore: integer listing the number of jobs that are ran on a
                        single core when using the fastr randomized search.
        use_fastr: Boolean determining of either fastr or joblib should be used
                   for the opimization.
        fastr_plugin: determines which plugin is used for fastr executions.
                When None, uses the default plugin from the fastr config.

    Returns:
        random_search: sklearn randomsearch object containing the results.
    """

    random_seed = np.random.randint(1, 5000)
    random_state = check_random_state(random_seed)

    regressors = ['SVR', 'RFR', 'SGDR', 'Lasso', 'ElasticNet']
    if any(clf in regressors for clf in param_grid['classifiers']):
        # We cannot do a stratified shuffle split with regression
        cv = ShuffleSplit(n_splits=5, test_size=test_size,
                          random_state=random_state)
    else:
        cv = StratifiedShuffleSplit(n_splits=5, test_size=test_size,
                                    random_state=random_state)

    if use_fastr:
        random_search = RandomizedSearchCVfastr(param_distributions=param_grid,
                                                n_iter=N_iter,
                                                scoring=scoring_method,
                                                n_jobs=n_cores,
                                                n_jobspercore=n_jobspercore,
                                                verbose=1, cv=cv,
                                                fastr_plugin=fastr_plugin)
    else:
        random_search = RandomizedSearchCVJoblib(param_distributions=param_grid,
                                                 n_iter=N_iter,
                                                 scoring=scoring_method,
                                                 n_jobs=n_cores,
                                                 verbose=1, cv=cv)
    random_search.fit(features, labels)
    print("Best found parameters:")
    print(random_search.best_params_)
    print("Best score using best parameters:")
    print(random_search.best_score_)

    return random_search
