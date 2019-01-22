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

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from PREDICT.featureselection.SelectGroups import SelectGroups
from sklearn.model_selection._validation import _fit_and_score
from PREDICT.featureselection.VarianceThreshold import selfeat_variance
from PREDICT.featureselection.StatisticalTestThreshold import StatisticalTestThreshold
import numpy as np
from sklearn.linear_model import Lasso
from sklearn.feature_selection import SelectFromModel
import scipy
from sklearn.preprocessing import Imputer
from sklearn.decomposition import PCA
from PREDICT.featureselection.Relief import SelectMulticlassRelief
from sklearn.multiclass import OneVsRestClassifier
from PREDICT.classification.estimators import RankedSVM


def fit_and_score(estimator, X, y, scorer,
                  train, test, para,
                  fit_params=None,
                  return_train_score=True,
                  return_n_test_samples=True,
                  return_times=True, return_parameters=True,
                  error_score='raise', verbose=True,
                  return_all=True):
    '''
    Fit an estimator to a dataset and score the performance. The following
    methods can currently be applied as preprocessing before fitting, in
    this order:
    1. Select features based on feature type group (e.g. shape, histogram).
    2. Apply feature imputation (WIP).
    3. Apply feature selection based on variance of feature among patients.
    4. Univariate statistical testing (e.g. t-test, Wilcoxon).
    5. Scale features with e.g. z-scoring.
    6. Use Relief feature selection.
    7. Select features based on a fit with a LASSO model.
    8. Select features using PCA.
    9. If a SingleLabel classifier is used for a MultiLabel problem,
        a OneVsRestClassifier is employed around it.

    All of the steps are optional.

    Parameters
    ----------
    estimator: sklearn estimator, mandatory
            Unfitted estimator which will be fit.

    X: array, mandatory
            Array containingfor each object (rows) the feature values
            (1st Column) and the associated feature label (2nd Column).

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

    return_all: boolean, default=True
            If False, only the ret object containing the performance will be
            returned. If True, the ret object plus all fitted objects will be
            returned.

    Returns
    ----------
    Depending on the return_all input parameter, either only ret or all objects
    below are returned.

    ret: list
        Contains optionally the train_scores and the test_scores,
        test_sample_counts, fit_time, score_time, parameters_est
        and parameters_all.

    GroupSel: PREDICT GroupSel Object
        Either None if the GroupSelFitted GroupSel Object.

    VarSel: PREDICT GroupSel Object
        Either None if the GroupSelFitted GroupSel Object.

    SelectModel: PREDICT GroupSel Object
        Either None if the GroupSelFitted GroupSel Object.

    feature_labels

    scaler

    imputer: PREDICT GroupSel Object
        Either None if the GroupSelFitted GroupSel Object.

    pca: PREDICT GroupSel Object
        Either None if the GroupSelFitted GroupSel Object.

    StatisticalSel: PREDICT GroupSel Object
        Either None if the GroupSelFitted GroupSel Object.

    ReliefSel: PREDICT GroupSel Object
        Either None if the GroupSelFitted GroupSel Object.

    '''
    # We copy the parameter object so we can alter it and keep the original
    para_estimator = para.copy()

    # X is a tuple: split in two arrays
    feature_values = np.asarray([x[0] for x in X])
    feature_labels = np.asarray([x[1] for x in X])

    # ------------------------------------------------------------------------
    # Groupwise feature selection
    if 'SelectGroups' in para_estimator:
        if verbose:
            print("Selecting groups of features.")
        del para_estimator['SelectGroups']
        # TODO: more elegant way to solve this
        feature_groups = ["histogram_features", "orientation_features",
                          "patient_features", "semantic_features",
                          "shape_features",
                          "coliage_features", 'vessel_features',
                          "phase_features", "log_features",
                          "texture_gabor_features", "texture_glcm_features",
                          "texture_glcmms_features", "texture_glrlm_features",
                          "texture_glszm_features", "texture_ngtdm_features",
                          "texture_lbp_features"]

        # Backwards compatability
        if 'texture_features' in para_estimator.keys():
            feature_groups.append('texture_features')

        # Check per feature group if the parameter is present
        parameters_featsel = dict()
        for group in feature_groups:
            if group not in para_estimator:
                # Default: do use the group, except for texture features
                if group == 'texture_features':
                    value = 'False'
                else:
                    value = 'True'
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

    # Delete the object if we do not need to return it
    if not return_all:
        del GroupSel

    # Check whether there are any features left
    if len(feature_values[0]) == 0:
        # TODO: Make a specific PREDICT exception for this warning.
        if verbose:
            print('[WARNING]: No features are selected! Probably all feature groups were set to False. Parameters:')
            print para

        # Return a zero performance dummy
        VarSel = None
        scaler = None
        SelectModel = None
        pca = None
        StatisticalSel = None
        imputer = None
        ReliefSel = None

        # Delete the non-used fields
        para_estimator = delete_nonestimator_parameters(para_estimator)

        ret = [0, 0, 0, 0, 0, para_estimator, para]
        if return_all:
            return ret, GroupSel, VarSel, SelectModel, feature_labels[0], scaler, imputer, pca, StatisticalSel, ReliefSel
        else:
            return ret

    # ------------------------------------------------------------------------
    # Feature imputation
    if 'Imputation' in para_estimator.keys():
        if para_estimator['Imputation'] == 'True':
            imp_type = para_estimator['ImputationMethod']
            imp_nn = para_estimator['ImputationNeighbours']

            imputer = Imputer(missing_values='NaN', strategy=imp_type,
                              n_neighbors=imp_nn, axis=0)
            imputer.fit(feature_values)
            feature_values = imputer.transform(feature_values)
        else:
            imputer = None
    else:
        imputer = None

    if 'Imputation' in para_estimator.keys():
        del para_estimator['Imputation']
        del para_estimator['ImputationMethod']
        del para_estimator['ImputationNeighbours']

    # Delete the object if we do not need to return it
    if not return_all:
        del imputer

    # ------------------------------------------------------------------------
    # FIXME: When only using LBP feature, X is 3 dimensional with 3rd dimension length 1
    if len(feature_values.shape) == 3:
        feature_values = np.reshape(feature_values, (feature_values.shape[0], feature_values.shape[1]))
    if len(feature_labels.shape) == 3:
        feature_labels = np.reshape(feature_labels, (feature_labels.shape[0], feature_labels.shape[1]))

    # Remove any NaN feature values if these are still left after imputation
    feature_values = replacenan(feature_values, verbose=verbose, feature_labels=feature_labels[0])

    # --------------------------------------------------------------------
    # Feature selection based on variance
    if para_estimator['Featsel_Variance'] == 'True':
        if verbose:
            print("Selecting features based on variance.")
        if verbose:
            print("Original Length: " + str(len(feature_values[0])))
        try:
            feature_values, feature_labels, VarSel =\
                selfeat_variance(feature_values, feature_labels)
        except ValueError:
            if verbose:
                print('[WARNING]: No features meet the selected Variance threshold! Skipping selection.')
            VarSel = None
        if verbose:
            print("New Length: " + str(len(feature_values[0])))
    else:
        VarSel = None
    del para_estimator['Featsel_Variance']

    # Delete the object if we do not need to return it
    if not return_all:
        del VarSel

    # Check whether there are any features left
    if len(feature_values[0]) == 0:
        # TODO: Make a specific PREDICT exception for this warning.
        if verbose:
            print('[WARNING]: No features are selected! Probably you selected a feature group that is not in your feature file. Parameters:')
            print para
        para_estimator = delete_nonestimator_parameters(para_estimator)

        # Return a zero performance dummy
        scaler = None
        SelectModel = None
        pca = None
        StatisticalSel = None
        ret = [0, 0, 0, 0, 0, para_estimator, para]
        if return_all:
            return ret, GroupSel, VarSel, SelectModel, feature_labels[0], scaler, imputer, pca, StatisticalSel, ReliefSel
        else:
            return ret

    # --------------------------------------------------------------------
    # Feature selection based on a statistical test
    if 'StatisticalTestUse' in para_estimator.keys():
        if para_estimator['StatisticalTestUse'] == 'True':
            metric = para_estimator['StatisticalTestMetric']
            threshold = para_estimator['StatisticalTestThreshold']
            if verbose:
                print("Selecting features based on statistical test. Method {}, threshold {}.").format(metric, str(round(threshold, 2)))
            if verbose:
                print("Original Length: " + str(len(feature_values[0])))

            StatisticalSel = StatisticalTestThreshold(metric=metric,
                                                      threshold=threshold)

            StatisticalSel.fit(feature_values, y)
            feature_values = StatisticalSel.transform(feature_values)
            feature_labels = StatisticalSel.transform(feature_labels)
            if verbose:
                print("New Length: " + str(len(feature_values[0])))
        else:
            StatisticalSel = None
        del para_estimator['StatisticalTestUse']
        del para_estimator['StatisticalTestMetric']
        del para_estimator['StatisticalTestThreshold']
    else:
        StatisticalSel = None

    # Delete the object if we do not need to return it
    if not return_all:
        del StatisticalSel

    # Check whether there are any features left
    if len(feature_values[0]) == 0:
        # TODO: Make a specific PREDICT exception for this warning.
        if verbose:
            print('[WARNING]: No features are selected! Probably you selected a feature group that is not in your feature file. Parameters:')
            print para
        para_estimator = delete_nonestimator_parameters(para_estimator)

        # Return a zero performance dummy
        scaler = None
        SelectModel = None
        pca = None
        ret = [0, 0, 0, 0, 0, para_estimator, para]
        if return_all:
            return ret, GroupSel, VarSel, SelectModel, feature_labels[0], scaler, imputer, pca, StatisticalSel, ReliefSel
        else:
            return ret

    # ------------------------------------------------------------------------
    # Feature scaling
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

    # Delete the object if we do not need to return it
    if not return_all:
        del scaler

    # --------------------------------------------------------------------
    # Relief feature selection, possibly multi classself.
    # Needs to be done after scaling!
    # para_estimator['ReliefUse'] = 'True'
    if 'ReliefUse' in para_estimator.keys():
        if para_estimator['ReliefUse'] == 'True':
            if verbose:
                print("Selecting features using relief.")

            # Get parameters from para_estimator
            n_neighbours = para_estimator['ReliefNN']
            sample_size = para_estimator['ReliefSampleSize']
            distance_p = para_estimator['ReliefDistanceP']
            numf = para_estimator['ReliefNumFeatures']

            ReliefSel = SelectMulticlassRelief(n_neighbours=n_neighbours,
                                               sample_size=sample_size,
                                               distance_p=distance_p,
                                               numf=numf)
            ReliefSel.fit(feature_values, y)
            if verbose:
                print("Original Length: " + str(len(feature_values[0])))
            feature_values = ReliefSel.transform(feature_values)
            if verbose:
                print("New Length: " + str(len(feature_values[0])))
            feature_labels = ReliefSel.transform(feature_labels)
        else:
            ReliefSel = None
    else:
        ReliefSel = None

    # Delete the object if we do not need to return it
    if not return_all:
        del ReliefSel

    if 'ReliefUse' in para_estimator.keys():
        del para_estimator['ReliefUse']
        del para_estimator['ReliefNN']
        del para_estimator['ReliefSampleSize']
        del para_estimator['ReliefDistanceP']
        del para_estimator['ReliefNumFeatures']

    # ------------------------------------------------------------------------
    # Perform feature selection using a model
    if 'SelectFromModel' in para_estimator.keys() and para_estimator['SelectFromModel'] == 'True':
        if verbose:
            print("Selecting features using lasso model.")
        # Use lasso model for feature selection

        # First, draw a random value for alpha and the penalty ratio
        alpha = scipy.stats.uniform(loc=0.0, scale=1.5).rvs()
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
    if 'SelectFromModel' in para_estimator.keys():
        del para_estimator['SelectFromModel']

    # Delete the object if we do not need to return it
    if not return_all:
        del SelectModel

    # ----------------------------------------------------------------
    # PCA dimensionality reduction
    # Principle Component Analysis
    if 'UsePCA' in para_estimator.keys() and para_estimator['UsePCA'] == 'True':
        if verbose:
            print('Fitting PCA')
            print("Original Length: " + str(len(feature_values[0])))
        if para_estimator['PCAType'] == '95variance':
            # Select first X components that describe 95 percent of the explained variance
            pca = PCA(n_components=None)
            pca.fit(feature_values)
            evariance = pca.explained_variance_ratio_
            num = 0
            sum = 0
            while sum < 0.95:
                sum += evariance[num]
                num += 1

            # Make a PCA based on the determined amound of components
            pca = PCA(n_components=num)
            pca.fit(feature_values)
            feature_values = pca.transform(feature_values)

        else:
            # Assume a fixed number of components
            n_components = int(para_estimator['PCAType'])
            pca = PCA(n_components=n_components)
            pca.fit(feature_values)
            feature_values = pca.transform(feature_values)

        if verbose:
            print("New Length: " + str(len(feature_values[0])))
    else:
        pca = None

    # Delete the object if we do not need to return it
    if not return_all:
        del pca

    if 'UsePCA' in para_estimator.keys():
        del para_estimator['UsePCA']
        del para_estimator['PCAType']

    # ----------------------------------------------------------------
    # Fitting and scoring
    # Only when using fastr this is an entry
    if 'Number' in para_estimator.keys():
        del para_estimator['Number']

    # For certainty, we delete all parameters again
    para_estimator = delete_nonestimator_parameters(para_estimator)

    # NOTE: This just has to go to the construct classifier function,
    # although it is more convenient here due to the hyperparameter search
    if type(y) is list:
        labellength = 1
    else:
        try:
            labellength = y.shape[1]
        except IndexError:
            labellength = 1

    if labellength > 1 and type(estimator) != RankedSVM:
        # Multiclass, hence employ a multiclass classifier for e.g. SVM, RF
        estimator.set_params(**para_estimator)
        estimator = OneVsRestClassifier(estimator)
        para_estimator = {}

    if verbose:
        print("Fitting ML.")
    ret = _fit_and_score(estimator, feature_values, y,
                         scorer, train,
                         test, verbose,
                         para_estimator, fit_params, return_train_score,
                         return_parameters,
                         return_n_test_samples,
                         return_times, error_score)

    # Paste original parameters in performance
    ret.append(para)

    if return_all:
        return ret, GroupSel, VarSel, SelectModel, feature_labels[0], scaler, imputer, pca, StatisticalSel, ReliefSel
    else:
        return ret


def delete_nonestimator_parameters(parameters):
    '''
    Delete all parameters in a parameter dictionary that are not used for the
    actual estimator.
    '''
    if 'Number' in parameters.keys():
        del parameters['Number']

    if 'UsePCA' in parameters.keys():
        del parameters['UsePCA']
        del parameters['PCAType']

    if 'Imputation' in parameters.keys():
        del parameters['Imputation']
        del parameters['ImputationMethod']
        del parameters['ImputationNeighbours']

    if 'SelectFromModel' in parameters.keys():
        del parameters['SelectFromModel']

    if 'Featsel_Variance' in parameters.keys():
        del parameters['Featsel_Variance']

    if 'FeatureScaling' in parameters.keys():
        del parameters['FeatureScaling']

    if 'StatisticalTestUse' in parameters.keys():
        del parameters['StatisticalTestUse']
        del parameters['StatisticalTestMetric']
        del parameters['StatisticalTestThreshold']

    return parameters


def replacenan(image_features, verbose=True, feature_labels=None):
    '''
    Replace the NaNs in an image feature matrix.
    '''
    image_features_temp = image_features.copy()
    for pnum, x in enumerate(image_features_temp):
        for fnum, value in enumerate(x):
            if np.isnan(value):
                if verbose:
                    if feature_labels is not None:
                        print("[PREDICT WARNING] NaN found, patient {}, label {}. Replacing with zero.").format(pnum, feature_labels[fnum])
                    else:
                        print("[PREDICT WARNING] NaN found, patient {}, label {}. Replacing with zero.").format(pnum, fnum)
                # Note: X is a list of lists, hence we cannot index the element directly
                image_features_temp[pnum, fnum] = 0

    return image_features_temp
