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
import scipy.stats as st
from scipy.special import logit, expit


def compute_confidence(metric, N_train, N_test, alpha=0.95):
    """
    Function to calculate the adjusted confidence interval
    metric: numpy array containing the result for a metric for the different cross validations
    (e.g. If 20 cross-validations are performed it is a list of length 20 with the calculated accuracy for
    each cross validation)
    N_train: Integer, number of training samples
    N_test: Integer, number of test_samples
    alpha: float ranging from 0 to 1 to calculate the alpha*100% CI, default 95%
    """

    # Convert to floats, as python 2 rounds the divisions if we have integers
    N_train = float(N_train)
    N_test = float(N_test)
    N_iterations = float(len(metric))

    if N_iterations == 1.0:
        print('[PREDICT Warning] Cannot compute a confidence interval for a single iteration.')
        print('[PREDICT Warning] CI will be set to value of single iteration.')
        metric_average = np.mean(metric)
        CI = (metric_average, metric_average)
    else:
        metric_average = np.mean(metric)
        S_uj = 1.0 / (N_iterations - 1) * np.sum((metric_average - metric)**2.0)

        metric_std = np.sqrt((1.0/N_iterations + N_test/N_train)*S_uj)

        CI = st.t.interval(alpha, N_iterations-1, loc=metric_average, scale=metric_std)

    return CI


def compute_confidence_logit(metric, N_train, N_test, alpha=0.95):
    """
    Function to calculate the adjusted confidence interval
    metric: numpy array containing the result for a metric for the different cross validations
    (e.g. If 20 cross-validations are performed it is a list of length 20 with the calculated accuracy for
    each cross validation)
    N_train: Integer, number of training samples
    N_test: Integer, number of test_samples
    alpha: float ranging from 0 to 1 to calculate the alpha*100% CI, default 95%
    """
    N_iterations = len(metric)

    # Compute average of logit function
    # metric_logit = [logit(x) for x in metric]
    logit_average = logit(np.mean(metric))

    # Compute metric average and corrected resampled t-test metric std
    metric_average = np.mean(metric)
    S_uj = 1.0 / (N_iterations - 1) * np.sum((metric_average - metric)**2.0)
    metric_std = np.sqrt((1.0/N_iterations + N_test/N_train)*S_uj)

    # Compute z(1-alpha/2) quantile
    q1 = 1.0-(1-alpha)/2
    z = st.t.ppf(q1, N_iterations - 1)

    # Compute logit confidence intervals according to Barbiero
    theta_L = logit_average - z * metric_std/(metric_average*(1 - metric_average))
    theta_U = logit_average + z * metric_std/(metric_average*(1 - metric_average))

    # Transform back
    CI = (expit(theta_L), expit(theta_U))

    return CI
