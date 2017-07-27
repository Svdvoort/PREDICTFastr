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

from sklearn.preprocessing import StandardScaler
from PREDICT.featureselection.SelectGroups import SelectGroups
from sklearn.model_selection._validation import _fit_and_score
from PREDICT.featureselection.selfeat import selfeat_variance
import numpy as np


def fit_and_score(estimator, X, y, scorer,
                  train, test, verbose, para,
                  fit_params,
                  return_train_score,
                  return_n_test_samples,
                  return_times, return_parameters,
                  error_score):

    para_estimator = para.copy()

    # X is a tuple: split in two arrays
    feature_values = np.asarray([x[0] for x in X])
    feature_labels = np.asarray([x[1] for x in X])

    # Perform feature selection if required
    if 'SelectGroups' in para_estimator:
        del para_estimator['SelectGroups']
        # TODO: more elegant way to solve this
        feature_groups = ["histogram_features", "orientation_features",
                          "patient_features", "semantic_features",
                          "shape_features", "texture_features",
                          "coliage_features"]
        parameters_featsel = dict()
        for group in feature_groups:
            if group not in para_estimator:
                # Default: do use the group
                value = True
            else:
                value = para_estimator[group]
                del para_estimator[group]

            parameters_featsel[group] = value

        GroupSel = SelectGroups(parameters=parameters_featsel)
        GroupSel.fit(feature_labels[0])
        feature_values = GroupSel.transform(feature_values)
        feature_labels = GroupSel.transform(feature_labels)
    else:
        GroupSel = None

    if len(feature_values[0]) == 0:
        # TODO: Make a specific PREDICT exception for this warning.
        print('[WARNING]: No features are selected! Probably all feature groups were set to False. Parameters:')
        print para

        # Return a zero performance dummy
        VarSel = None
        scaler = None
        ret = [0, 0, 0, 0, 0, para_estimator, para]
    else:
        # FIXME: When only ysing LBP feature, X is 3 dimensional with 3rd dimension length 1
        if len(feature_values.shape) == 3:
            feature_values = np.reshape(feature_values, (feature_values.shape[0], feature_values.shape[1]))
        if len(feature_labels.shape) == 3:
            feature_labels = np.reshape(feature_labels, (feature_labels.shape[0], feature_labels.shape[1]))

        if para_estimator['Featsel_Variance'] == 'True':
            feature_values, feature_labels, VarSel =\
             selfeat_variance(feature_values, feature_labels)
        else:
            VarSel = None
        del para_estimator['Featsel_Variance']

        # Fit and score the classifier
        if len(feature_values[0]) == 0:
            # TODO: Make a specific PREDICT exception for this warning.
            print('[WARNING]: No features are selected! Probably you selected a feature group that is not in your feature file. Parameters:')
            print para

            # Return a zero performance dummy
            scaler = None
            ret = [0, 0, 0, 0, 0, para_estimator, para]
        else:
            scaler = StandardScaler().fit(feature_values)
            feature_values = scaler.transform(feature_values)

            # Only when using fastr this is an entry
            if 'Number' in para_estimator.keys():
                del para_estimator['Number']

            ret = _fit_and_score(estimator, feature_values, y,
                                 scorer, train,
                                 test, verbose,
                                 para_estimator, fit_params, return_train_score,
                                 return_parameters,
                                 return_n_test_samples,
                                 return_times, error_score)

            # Paste original parameters in performance
            ret.append(para)

    return ret, GroupSel, VarSel, feature_labels[0], scaler
