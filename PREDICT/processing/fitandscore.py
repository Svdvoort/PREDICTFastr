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
from sklearn.preprocessing import MinMaxScaler
from PREDICT.featureselection.SelectGroups import SelectGroups
from sklearn.model_selection._validation import _fit_and_score
from PREDICT.featureselection.selfeat import selfeat_variance
import numpy as np
from sklearn.linear_model import Lasso
from sklearn.feature_selection import SelectFromModel
import scipy


def fit_and_score(estimator, X, y, scorer,
                  train, test, para,
                  fit_params=None,
                  return_train_score=True,
                  return_n_test_samples=True,
                  return_times=True, return_parameters=True,
                  error_score='raise', verbose=True):
    '''
    Fit an estimator to a dataset and score the performance. The following
    methods can currently be applied as preprocessing before fitting in
    this order:
    1. Apply feature selection based on type group.
    2. Apply feature selection based on variance of feature among patients.
    3. Scale features with e.g. z-scoring.

    Parameters
    ----------
    estimator: sklearn estimator, mandatory
            Unfitted estimator which will be fit.

    X: array, mandatory
            Array containing the feature values (columns) for each object (rows).

    y: list(?), mandatory
            List containing the labels of the objects.

    scorer: sklearn scorer, mandatory
            Function used as optimization criterion for the hyperparamater optimization.

    train: list, mandatory
            Indices of the objects to be used as training set.

    test: list, mandatory
            Indices of the objects to be used as testing set.

    para: dictionary, mandatory
            Contains the settings used for the above preprocessing functions
            and the fitting. TODO: Create a default object and show the
            fields.

    fit_params:dictionary, default None
            Parameters supplied to the estimator for fitting. See the SKlearn
            site for the parameters of the estimators.

    return_train_score: boolean, default True
            Save the training score to the final SearchCV object.

    return_n_test_samples: boolean, default True
            Save the number of times each sample was used in the test set
            to the final SearchCV object.

    return_times: boolean, default True
            Save the time spend for each fit to the final SearchCV object.

    return_parameters: boolean, default True
            Return the parameters used in the final fit to the final SearchCV
            object.

    error_score: numeric or "raise" by default
            Value to assign to the score if an error occurs in estimator
            fitting. If set to "raise", the error is raised. If a numeric
            value is given, FitFailedWarning is raised. This parameter
            does not affect the refit step, which will always raise the error.

    verbose: boolean, default=True
            If True, print intermediate progress to command line. Warnings are
            always printed.

    '''
    # We copy the parameter object so we can alter it and keep the original
    para_estimator = para.copy()

    # X is a tuple: split in two arrays
    feature_values = np.asarray([x[0] for x in X])
    feature_labels = np.asarray([x[1] for x in X])

    # Perform feature selection if required
    if 'SelectGroups' in para_estimator:
        if verbose:
            print("Selecting groups of features.")
        del para_estimator['SelectGroups']
        # TODO: more elegant way to solve this
        feature_groups = ["histogram_features", "orientation_features",
                          "patient_features", "semantic_features",
                          "shape_features", "texture_features",
                          "coliage_features", 'vessel_features',
                          "phase_features", "log_features"]
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
        if verbose:
            print("Original Length: " + str(len(feature_values[0])))
        feature_values = GroupSel.transform(feature_values)
        if verbose:
            print("New Length: " + str(len(feature_values[0])))
        feature_labels = GroupSel.transform(feature_labels)
    else:
        GroupSel = None

    # Perform feature selection using a model
    para_estimator['SelectFromModel'] = False
    if para_estimator['SelectFromModel']:
        if verbose:
            print("Selecting features using lasso model.")
        # Use lasso model for feature selection

        # First, draw a random value for alpha and the penalty ratio
        alpha = scipy.stats.uniform(loc=1.0, scale=0.5).rvs()
        # l1_ratio = scipy.stats.uniform(loc=0.5, scale=0.4).rvs()

        # Create and fit lasso model
        lassomodel = Lasso(alpha=alpha)
        lassomodel.fit(feature_values, y)

        # Use fit to select optimal features
        SelectModel = SelectFromModel(lassomodel, prefit=True)
        if verbose:
            print("Original Length: " + str(len(feature_values[0])))
        feature_values = SelectModel.transform(feature_values)
        if verbose:
            print("New Length: " + str(len(feature_values[0])))
        feature_labels = SelectModel.transform(feature_labels)
    else:
        SelectModel = None
    del para_estimator['SelectFromModel']

    if len(feature_values[0]) == 0:
        # TODO: Make a specific PREDICT exception for this warning.
        print('[WARNING]: No features are selected! Probably all feature groups were set to False. Parameters:')
        print para

        # Return a zero performance dummy
        VarSel = None
        scaler = None

        # Delete the non-used fields
        if 'Featsel_Variance' in para_estimator.keys():
            del para_estimator['Featsel_Variance']
        if 'FeatureScaling' in para_estimator.keys():
            del para_estimator['FeatureScaling']

        ret = [0, 0, 0, 0, 0, para_estimator, para]
    else:
        # FIXME: When only using LBP feature, X is 3 dimensional with 3rd dimension length 1
        if len(feature_values.shape) == 3:
            feature_values = np.reshape(feature_values, (feature_values.shape[0], feature_values.shape[1]))
        if len(feature_labels.shape) == 3:
            feature_labels = np.reshape(feature_labels, (feature_labels.shape[0], feature_labels.shape[1]))

        if para_estimator['Featsel_Variance'] == 'True':
            if verbose:
                print("Selecting features based on variance.")
            if verbose:
                print("Original Length: " + str(len(feature_values[0])))
            try:
                feature_values, feature_labels, VarSel =\
                    selfeat_variance(feature_values, feature_labels)
            except ValueError:
                print('[WARNING]: No features meet the selected Variance threshold! Skipping selection.')
                VarSel = None
            if verbose:
                print("New Length: " + str(len(feature_values[0])))
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
            if 'FeatureScaling' in para_estimator:
                if verbose:
                    print("Fitting scaler and transforming features.")
                if para_estimator['FeatureScaling'] == 'z_score':
                    scaler = StandardScaler().fit(feature_values)
                elif para_estimator['FeatureScaling'] == 'minmax':
                    scaler = MinMaxScaler().fit(feature_values)
                else:
                    scaler = None

                if scaler is not None:
                    feature_values = scaler.transform(feature_values)
                del para_estimator['FeatureScaling']
            else:
                scaler = None

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

    return ret, GroupSel, VarSel, SelectModel, feature_labels[0], scaler
