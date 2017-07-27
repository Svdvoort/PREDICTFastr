#!/usr/bin/env python

# Copyright 2011-2017 Biomedical Imaging Group Rotterdam, Departments of
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
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import RandomizedSearchCV

import PREDICT.IOparser.config_general as config_io
from PREDICT.processing.RandomizedSearchCVfastr import RandomizedSearchCVfastr


def random_search_parameters(features, labels, N_iter, test_size,
                             classifier, param_grid, scoring_method,
                             score_threshold, use_fastr=False):

    random_seed = np.random.randint(1, 5000)
    random_state = check_random_state(random_seed)

    cv = StratifiedShuffleSplit(n_splits=5, test_size=test_size,
                                random_state=random_state)

    config = config_io.load_config()
    n_jobs = config['Joblib']['njobs']

    if use_fastr:
        random_search = RandomizedSearchCVfastr(classifier, param_distributions=param_grid,
                                                n_iter=N_iter, scoring=scoring_method, n_jobs=n_jobs,
                                                verbose=1, cv=cv)
    else:
        random_search = RandomizedSearchCV(classifier, param_distributions=param_grid,
                                           n_iter=N_iter, scoring=scoring_method, n_jobs=n_jobs,
                                           verbose=1, cv=cv)
    random_search.fit(features, labels)
    print(random_search.best_score_)
    print(random_search.best_params_)

    return random_search
