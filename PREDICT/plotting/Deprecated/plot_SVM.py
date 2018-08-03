#!/usr/bin/env python

# Copyright 2011-2018 Biomedical Imaging Group Rotterdam, Departments of
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
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
import sys
import compute_CI
import pandas as pd
import os
import natsort
import collections
import sklearn

import PREDICT.genetics.genetic_processing as gp
from sklearn.base import clone


def plot_single_SVM(prediction, label_data, label_type, show_plots=False,
                    key=None, alpha=0.95, ensemble=False):
    '''
    Plot the output of a single binary estimator, e.g. a SVM.

    Parameters
    ----------
    prediction: pandas dataframe or string, mandatory
                output of trainclassifier function, either a pandas dataframe
                or a HDF5 file

    label_data: string, mandatory
            Contains the path referring to a .txt file containing the
            patient label(s) and value(s) to be used for learning. See
            the Github Wiki for the format.

    label_type: string, mandatory
            Name of the label to extract from the label data to test the
            estimator on.

    show_plots: Boolean, default False
            Determine whether matplotlib performance plots are made.

    key: string, default None
            As the prediction object can contain multiple estimators,
            the key is used to select the desired estimator. If None, the
            first key and estimator will be used

    alpha: float, default 0.95
            Significance of confidence intervals.

    '''

    # Load the prediction object if it's a hdf5 file
    if type(prediction) is not pd.core.frame.DataFrame:
        if os.path.isfile(prediction):
            prediction = pd.read_hdf(prediction)

    # Select the estimator from the pandas dataframe to use
    keys = prediction.keys()
    SVMs = list()
    if key is None:
        label = keys[0]
    else:
        label = key

    # Extract the estimators, features and labels
    SVMs = prediction[label]['classifiers']
    Y_test = prediction[label]['Y_test']
    X_test = prediction[label]['X_test']
    X_train = prediction[label]['X_train']
    Y_train = prediction[label]['Y_train']
    feature_labels = prediction[label]['feature_labels']

    # Load the label data
    if type(label_data) is not dict:
        if os.path.isfile(label_data):
            label_data = gp.load_mutation_status(label_data, [[label_type]])

    patient_IDs = label_data['patient_IDs']
    mutation_label = label_data['mutation_label']

    # Create lists for performance measures
    N_iterations = float(len(SVMs))
    sensitivity = list()
    specificity = list()
    precision = list()
    accuracy = list()
    auc = list()
    f1_score_list = list()
    patient_classification_list = dict()

    # Loop over the test sets, which probably correspond with cross validation
    # iterations
    for i in range(0, len(Y_test)):
        print("Cross validation {} / {}.").format(str(i + 1), str(len(Y_test)))
        test_patient_IDs = prediction[label]['patient_ID_test'][i]
        train_patient_IDs = prediction[label]['patient_ID_train'][i]
        X_test_temp = X_test[i]
        X_train_temp = X_train[i]
        Y_train_temp = Y_train[i]
        Y_test_temp = Y_test[i]
        test_indices = list()

        # Check which patients are in the test set.
        for i_ID in test_patient_IDs:
            test_indices.append(np.where(patient_IDs == i_ID)[0][0])

            # Initiate counting how many times a patient is classified correctly
            if i_ID not in patient_classification_list:
                patient_classification_list[i_ID] = dict()
                patient_classification_list[i_ID]['N_test'] = 0
                patient_classification_list[i_ID]['N_correct'] = 0
                patient_classification_list[i_ID]['N_wrong'] = 0

            patient_classification_list[i_ID]['N_test'] += 1

        # Extract ground truth
        y_truth = Y_test_temp

        # If requested, first let the SearchCV object create an ensemble
        if ensemble:
            # NOTE: Remove these two lines
            cv_iter = list(SVMs[i].cv.split(X_train_temp, Y_train_temp))
            SVMs[i].cv_iter = cv_iter

            X_train_temp = [(x, feature_labels) for x in X_train_temp]
            SVMs[i].create_ensemble(X_train_temp, Y_train_temp, verbose=True)

        # Create prediction
        y_prediction = SVMs[i].predict(X_test_temp)

        print "Truth: ", y_truth
        print "Prediction: ", y_prediction

        # Add if patient was classified correctly or not to counting
        for i_truth, i_predict, i_test_ID in zip(y_truth, y_prediction, test_patient_IDs):
            if i_truth == i_predict:
                patient_classification_list[i_test_ID]['N_correct'] += 1
            else:
                patient_classification_list[i_test_ID]['N_wrong'] += 1

        # Compute confusion matrix and use for sensitivity/specificity
        c_mat = confusion_matrix(y_truth, y_prediction)
        TN = c_mat[0, 0]
        FN = c_mat[1, 0]
        TP = c_mat[1, 1]
        FP = c_mat[0, 1]

        if FN == 0 and TP == 0:
            sensitivity.append(0)
        else:
            sensitivity.append(float(TP)/(TP+FN))
        if FP == 0 and TN == 0:
            specificity.append(0)
        else:
            specificity.append(float(TN)/(FP+TN))
        if TP == 0 and FP == 0:
            precision.append(0)
        else:
            precision.append(float(TP)/(TP+FP))

        # Additionally, compute accuracy, AUC and f1-score
        accuracy.append(accuracy_score(y_truth, y_prediction))
        y_score = SVMs[i].decision_function(X_test_temp)
        auc.append(roc_auc_score(y_truth, y_score))
        f1_score_list.append(f1_score(y_truth, y_prediction, average='weighted'))

    # Extract sample size
    N_1 = float(len(train_patient_IDs))
    N_2 = float(len(test_patient_IDs))

    # Compute alpha confidence intervallen
    stats = dict()
    stats["Accuracy 95%:"] = str(compute_CI.compute_confidence(accuracy, N_1, N_2, alpha))

    stats["AUC 95%:"] = str(compute_CI.compute_confidence(auc, N_1, N_2, alpha))

    stats["F1-score 95%:"] = str(compute_CI.compute_confidence(f1_score_list, N_1, N_2, alpha))

    stats["Precision 95%:"] = str(compute_CI.compute_confidence(precision, N_1, N_2, alpha))

    stats["Sensitivity 95%: "] = str(compute_CI.compute_confidence(sensitivity, N_1, N_2, alpha))

    stats["Specificity 95%:"] = str(compute_CI.compute_confidence(specificity, N_1, N_2, alpha))

    print("Accuracy 95%:" + str(compute_CI.compute_confidence(accuracy, N_1, N_2, alpha)))

    print("AUC 95%:" + str(compute_CI.compute_confidence(auc, N_1, N_2, alpha)))

    print("F1-score 95%:" + str(compute_CI.compute_confidence(f1_score_list, N_1, N_2, alpha)))

    print("Precision 95%:" + str(compute_CI.compute_confidence(precision, N_1, N_2, alpha)))

    print("Sensitivity 95%: " + str(compute_CI.compute_confidence(sensitivity, N_1, N_2, alpha)))

    print("Specificity 95%:" + str(compute_CI.compute_confidence(specificity, N_1, N_2, alpha)))

    # Extract statistics on how often patients got classified correctly
    alwaysright = dict()
    alwayswrong = dict()
    percentages = dict()
    for i_ID in patient_classification_list:
        percentage_right = patient_classification_list[i_ID]['N_correct'] / float(patient_classification_list[i_ID]['N_test'])

        label = mutation_label[0][np.where(i_ID == patient_IDs)]
        label = label[0][0]
        if percentage_right == 1.0:
            alwaysright[i_ID] = label
            print(("Always Right: {}, label {}").format(i_ID, label))

        elif percentage_right == 0:
            alwayswrong[i_ID] = label
            print(("Always Wrong: {}, label {}").format(i_ID, label))

        else:
            percentages[i_ID] = str(label) + ': ' + str(round(percentage_right, 2) * 100) + '%'

    stats["Always right"] = alwaysright
    stats["Always wrong"] = alwayswrong
    stats['Percentages'] = percentages

    if show_plots:
        # Plot some characteristics in boxplots
        import matplotlib.pyplot as plt

        plt.figure()
        plt.boxplot(accuracy)
        plt.ylim([-0.05, 1.05])
        plt.ylabel('Accuracy')
        plt.tick_params(
            axis='x',          # changes apply to the x-axis
            which='both',      # both major and minor ticks are affected
            bottom='off',      # ticks along the bottom edge are off
            top='off',         # ticks along the top edge are off
            labelbottom='off')  # labels along the bottom edge are off
        plt.tight_layout()
        plt.show()

        plt.figure()
        plt.boxplot(auc)
        plt.ylim([-0.05, 1.05])
        plt.ylabel('AUC')
        plt.tick_params(
            axis='x',          # changes apply to the x-axis
            which='both',      # both major and minor ticks are affected
            bottom='off',      # ticks along the bottom edge are off
            top='off',         # ticks along the top edge are off
            labelbottom='off')  # labels along the bottom edge are off
        plt.tight_layout()
        plt.show()

        plt.figure()
        plt.boxplot(precision)
        plt.ylim([-0.05, 1.05])
        plt.ylabel('Precision')
        plt.tick_params(
            axis='x',          # changes apply to the x-axis
            which='both',      # both major and minor ticks are affected
            bottom='off',      # ticks along the bottom edge are off
            top='off',         # ticks along the top edge are off
            labelbottom='off')  # labels along the bottom edge are off
        plt.tight_layout()
        plt.show()

        plt.figure()
        plt.boxplot(sensitivity)
        plt.ylim([-0.05, 1.05])
        plt.ylabel('Sensitivity')
        plt.tick_params(
            axis='x',          # changes apply to the x-axis
            which='both',      # both major and minor ticks are affected
            bottom='off',      # ticks along the bottom edge are off
            top='off',         # ticks along the top edge are off
            labelbottom='off')  # labels along the bottom edge are off
        plt.tight_layout()
        plt.show()

        plt.figure()
        plt.boxplot(specificity)
        plt.ylim([-0.05, 1.05])
        plt.ylabel('Specificity')
        plt.tick_params(
            axis='x',          # changes apply to the x-axis
            which='both',      # both major and minor ticks are affected
            bottom='off',      # ticks along the bottom edge are off
            top='off',         # ticks along the top edge are off
            labelbottom='off')  # labels along the bottom edge are off
        plt.tight_layout()
        plt.show()

    return stats


def plot_multi_SVM(prediction, label_data, label_type, show_plots=False,
                   key=None, n_classifiers=[1, 10, 50], alpha=0.95):
    '''
    Plot the output of a an ensemble of binary estimators, e.g. a SVMs.

    Parameters
    ----------
    prediction: pandas dataframe or string, mandatory
                output of trainclassifier function, either a pandas dataframe
                or a HDF5 file

    label_data: string, mandatory
            Contains the path referring to a .txt file containing the
            patient label(s) and value(s) to be used for learning. See
            the Github Wiki for the format.

    label_type: string, mandatory
            Name of the label to extract from the label data to test the
            estimator on.

    show_plots: Boolean, default False
            Determine whether matplotlib performance plots are made.

    key: string, default None
            As the prediction object can contain multiple estimators,
            the key is used to select the desired estimator. If None, the
            first key and estimator will be used

    alpha: float, default 0.95
            Significance of confidence intervals.

    '''
    # Load the prediction object if it's a hdf5 file
    if type(prediction) is not pd.core.frame.DataFrame:
        if os.path.isfile(prediction):
            prediction = pd.read_hdf(prediction)

    # Select the estimator from the pandas dataframe to use
    keys = prediction.keys()
    SVMs = list()
    if key is None:
        label = keys[0]
    else:
        label = key

    # Load the estmators, the test and training features, labels and patient IDs
    SVMs = prediction[label]['classifiers']
    N_iterations = float(len(SVMs))
    Y_test = prediction[label]['Y_test']
    X_test = prediction[label]['X_test']
    X_train = prediction[label]['X_train']
    Y_train = prediction[label]['Y_train']
    test_patient_IDs = prediction[label]['patient_ID_test']
    train_patient_IDs = prediction[label]['patient_ID_train']
    feature_labels = prediction[label]['feature_labels']

    # Load the label data
    if type(label_data) is not dict:
        if os.path.isfile(label_data):
            label_data = gp.load_mutation_status(label_data, [[label_type]])
    print label_data
    patient_IDs = label_data['patient_IDs']
    mutation_label = label_data['mutation_label']

    # Generate output for each ensemble consisting of n_classifiers
    stats = dict()
    predictions = dict()
    for n_class in n_classifiers:

        # Create listst for output performance
        sensitivity = list()
        specificity = list()
        precision = list()
        accuracy = list()
        auc = list()
        f1_score_list = list()
        patient_classification_list = dict()
        trained_classifiers = list()
        y_score = list()
        y_test = list()
        pid_test = list()
        y_predict = list()

        # Create empty score entries for each patient
        empty_scores = {k: '' for k in natsort.natsorted(patient_IDs)}
        empty_scores = collections.OrderedDict(sorted(empty_scores.items()))
        predictions[str(n_class)] = dict()
        predictions[str(n_class)]['y_score'] = list()
        predictions[str(n_class)]['y_test'] = list()

        # Loop over the estimators, which probably correspond with cross validation
        # iterations
        params = dict()
        for num, s in enumerate(SVMs):
            scores = empty_scores.copy()
            print("Processing {} / {}.").format(str(num + 1), str(len(SVMs)))
            trained_classifiers.append(s)

            # Extract test info
            test_patient_IDs_temp = test_patient_IDs[num]
            train_patient_IDs_temp = train_patient_IDs[num]
            X_train_temp = X_train[num]
            Y_train_temp = Y_train[num]
            X_test_temp = X_test[num]
            Y_test_temp = Y_test[num]

            # Stack feature values and labels together again in single object
            X_train_temp = [(x, feature_labels) for x in X_train_temp]

            # Extract sample size
            N_1 = float(len(train_patient_IDs_temp))
            N_2 = float(len(test_patient_IDs_temp))

            # Check which patients are in the test set.
            test_indices = list()
            for i_ID in test_patient_IDs_temp:
                test_indices.append(np.where(patient_IDs == i_ID)[0][0])

                # Initiate counting how many times a patient is classified correctly
                if i_ID not in patient_classification_list:
                    patient_classification_list[i_ID] = dict()
                    patient_classification_list[i_ID]['N_test'] = 0
                    patient_classification_list[i_ID]['N_correct'] = 0
                    patient_classification_list[i_ID]['N_wrong'] = 0

                patient_classification_list[i_ID]['N_test'] += 1

            # Get ground truth labels
            y_truth = Y_test_temp

            # Predict  labels using the top N classifiers
            results = s.cv_results_['rank_test_score']
            indices = range(0, len(results))
            sortedindices = [x for _, x in sorted(zip(results, indices))]
            sortedindices = sortedindices[0:n_class]
            y_prediction = np.zeros([n_class, len(y_truth)])
            y_score = np.zeros([n_class, len(y_truth)])

            # Get some base objects required
            y_train = Y_train_temp
            y_train_prediction = np.zeros([n_class, len(y_train)])
            train = np.asarray(range(0, len(y_train)))
            test = train  # This is in order to use the full training dataset to train the model

            # Loop over the sortedindice selected and refit an estimator for each setting
            # NOTE: need to build this in the SearchCVFastr Object
            for i, index in enumerate(sortedindices):
                print("Processing number {} of {} classifiers.").format(str(i + 1), str(n_class))
                X_testtemp = X_test_temp[:]
                s_temp = clone(s)

                # Get the parameters from the index
                parameters_est = s.cv_results_['params'][index]
                parameters_all = s.cv_results_['params_all'][index]

                # NOTE: kernel parameter can be unicode, which we fix
                kernel = str(parameters_est[u'kernel'])
                del parameters_est[u'kernel']
                del parameters_all[u'kernel']
                parameters_est['kernel'] = kernel
                parameters_all['kernel'] = kernel

                # Refit a classifier using the settings given
                print("Refitting classifier with best settings.")
                # Only when using fastr this is an entry, but needs to be deleted
                if 'Number' in parameters_est.keys():
                    del parameters_est['Number']

                # Refit object with selected settings
                s_temp.refit_and_score(X_train_temp, y_train, parameters_all,
                                       parameters_est, train, test)

                # Throw away the feature labels
                X_train_temp_eval = [x[0] for x in X_train_temp]

                # Predict the posterios using the fitted classifier for the training set
                print("Evaluating performance on training set.")
                if hasattr(s_temp, 'predict_proba'):
                    probabilities = s_temp.predict_proba(X_train_temp_eval)
                    y_train_prediction[i, :] = probabilities[:, 1]
                else:
                    # Regression has no probabilities
                    probabilities = s_temp.predict(X_train_temp_eval)
                    y_train_prediction[i, :] = probabilities[:]

                # Predict the posterios using the fitted classifier for the test set
                print("Evaluating performance on test set.")
                if hasattr(s_temp, 'predict_proba'):
                    probabilities = s_temp.predict_proba(X_testtemp)
                    y_prediction[i, :] = probabilities[:, 1]
                else:
                    # Regression has no probabilities
                    probabilities = s_temp.predict(X_testtemp)
                    y_prediction[i, :] = probabilities[:]

                # For a VM, we can compute a score per patient with the decision function
                if type(s.estimator) == sklearn.svm.classes.SVC:
                    y_score[i, :] = s_temp.decision_function(X_testtemp)
                else:
                    y_score[i, :] = s_temp.decision_function(X_testtemp)[:, 0]

                # Add the parameter settngs of this iteration to the collection object
                for k in parameters_all.keys():
                    if k not in params.keys():
                        params[k] = list()
                    params[k].append(parameters_all[k])

                # Save some memory
                del s_temp, parameters_est, parameters_all, probabilities

            # Take mean over posteriors of top n
            y_train_prediction_m = np.mean(y_train_prediction, axis=0)
            y_prediction_m = np.mean(y_prediction, axis=0)
            y_score = y_prediction_m

            if type(s.estimator) == sklearn.svm.classes.SVC:
                # Look for optimal F1 performance on training set to set threshold
                thresholds = np.arange(0, 1, 0.01)
                f1_scores = list()
                y_train_prediction = np.zeros(y_train_prediction_m.shape)
                for t in thresholds:
                    for ip, y in enumerate(y_train_prediction_m):
                        if y > t:
                            y_train_prediction[ip] = 1
                        else:
                            y_train_prediction[ip] = 0

                    f1_scores.append(f1_score(y_train_prediction, y_train, average='weighted'))

                # Use best threshold to determine test score
                best_index = np.argmax(f1_scores)
                best_thresh = thresholds[best_index]

                # NOTE: due to overfitting in past, we do not fit a threshold here after all.
                best_thresh = 0.5
                y_prediction = np.zeros(y_prediction_m.shape)
                for ip, y in enumerate(y_prediction_m):
                    if y > best_thresh:
                        y_prediction[ip] = 1
                    else:
                        y_prediction[ip] = 0

                y_prediction = [min(max(y, 0), 1) for y in y_prediction]
            else:
                y_prediction = y_prediction_m
                y_prediction = [min(max(y, 0), 1) for y in y_prediction]

            predictions[str(n_class)]['y_score'].append(y_score[:])
            predictions[str(n_class)]['y_test'].append(y_truth[:])
            print "Truth: ", y_truth
            print "Prediction: ", y_prediction

            # Add if patient was classified correctly or not to counting
            for i_truth, i_predict, i_test_ID in zip(y_truth, y_prediction, test_patient_IDs_temp):
                if i_truth == i_predict:
                    patient_classification_list[i_test_ID]['N_correct'] += 1
                else:
                    patient_classification_list[i_test_ID]['N_wrong'] += 1

            # Compute confusion matrix and use for sensitivity/specificity
            c_mat = confusion_matrix(y_truth, y_prediction)
            TN = c_mat[0, 0]
            FN = c_mat[1, 0]
            TP = c_mat[1, 1]
            FP = c_mat[0, 1]

            if FN == 0 and TP == 0:
                sensitivity.append(0)
            else:
                sensitivity.append(float(TP)/(TP+FN))
            if FP == 0 and TN == 0:
                specificity.append(0)
            else:
                specificity.append(float(TN)/(FP+TN))
            if TP == 0 and FP == 0:
                precision.append(0)
            else:
                precision.append(float(TP)/(TP+FP))

            # Additionally, compute accuracy, AUC and f1-score
            accuracy.append(accuracy_score(y_truth, y_prediction))
            auc.append(roc_auc_score(y_truth, y_score))
            f1_score_list.append(f1_score(y_truth, y_prediction, average='weighted'))

        # Compute alpha confidence intervallen
        stats[str(n_class)] = dict()
        stats[str(n_class)]["Accuracy 95%:"] = str(compute_CI.compute_confidence(accuracy, N_1, N_2, alpha))

        stats[str(n_class)]["AUC 95%:"] = str(compute_CI.compute_confidence(auc, N_1, N_2, alpha))

        stats[str(n_class)]["F1-score 95%:"] = str(compute_CI.compute_confidence(f1_score_list, N_1, N_2, alpha))

        stats[str(n_class)]["Precision 95%:"] = str(compute_CI.compute_confidence(precision, N_1, N_2, alpha))

        stats[str(n_class)]["Sensitivity 95%: "] = str(compute_CI.compute_confidence(sensitivity, N_1, N_2, alpha))

        stats[str(n_class)]["Specificity 95%:"] = str(compute_CI.compute_confidence(specificity, N_1, N_2, alpha))

        print("Accuracy 95%:" + str(compute_CI.compute_confidence(accuracy, N_1, N_2, alpha)))

        print("AUC 95%:" + str(compute_CI.compute_confidence(auc, N_1, N_2, alpha)))

        print("F1-score 95%:" + str(compute_CI.compute_confidence(f1_score_list, N_1, N_2, alpha)))

        print("Precision 95%:" + str(compute_CI.compute_confidence(precision, N_1, N_2, alpha)))

        print("Sensitivity 95%: " + str(compute_CI.compute_confidence(sensitivity, N_1, N_2, alpha)))

        print("Specificity 95%:" + str(compute_CI.compute_confidence(specificity, N_1, N_2, alpha)))

        # Extract statistics on how often patients got classified correctly
        alwaysright = dict()
        alwayswrong = dict()
        percentages = dict()
        for i_ID in patient_classification_list:
            percentage_right = patient_classification_list[i_ID]['N_correct'] / float(patient_classification_list[i_ID]['N_test'])

            label = mutation_label[0][np.where(i_ID == patient_IDs)]
            label = label[0][0]
            if percentage_right == 1.0:
                alwaysright[i_ID] = label
                print(("Always Right: {}, label {}").format(i_ID, label))

            elif percentage_right == 0:
                alwayswrong[i_ID] = label
                print(("Always Wrong: {}, label {}").format(i_ID, label))

            else:
                percentages[i_ID] = str(label) + ': ' + str(round(percentage_right, 2) * 100) + '%'

        stats[str(n_class)]["Always right"] = alwaysright
        stats[str(n_class)]["Always wrong"] = alwayswrong
        stats[str(n_class)]['Percentages'] = percentages

        if show_plots:
            # Plot some characteristics in boxplots
            import matplotlib.pyplot as plt

            plt.figure()
            plt.boxplot(accuracy)
            plt.ylim([-0.05, 1.05])
            plt.ylabel('Accuracy')
            plt.tick_params(
                axis='x',          # changes apply to the x-axis
                which='both',      # both major and minor ticks are affected
                bottom='off',      # ticks along the bottom edge are off
                top='off',         # ticks along the top edge are off
                labelbottom='off')  # labels along the bottom edge are off
            plt.tight_layout()
            plt.show()

            plt.figure()
            plt.boxplot(auc)
            plt.ylim([-0.05, 1.05])
            plt.ylabel('AUC')
            plt.tick_params(
                axis='x',          # changes apply to the x-axis
                which='both',      # both major and minor ticks are affected
                bottom='off',      # ticks along the bottom edge are off
                top='off',         # ticks along the top edge are off
                labelbottom='off')  # labels along the bottom edge are off
            plt.tight_layout()
            plt.show()

            plt.figure()
            plt.boxplot(precision)
            plt.ylim([-0.05, 1.05])
            plt.ylabel('Precision')
            plt.tick_params(
                axis='x',          # changes apply to the x-axis
                which='both',      # both major and minor ticks are affected
                bottom='off',      # ticks along the bottom edge are off
                top='off',         # ticks along the top edge are off
                labelbottom='off')  # labels along the bottom edge are off
            plt.tight_layout()
            plt.show()

            plt.figure()
            plt.boxplot(sensitivity)
            plt.ylim([-0.05, 1.05])
            plt.ylabel('Sensitivity')
            plt.tick_params(
                axis='x',          # changes apply to the x-axis
                which='both',      # both major and minor ticks are affected
                bottom='off',      # ticks along the bottom edge are off
                top='off',         # ticks along the top edge are off
                labelbottom='off')  # labels along the bottom edge are off
            plt.tight_layout()
            plt.show()

            plt.figure()
            plt.boxplot(specificity)
            plt.ylim([-0.05, 1.05])
            plt.ylabel('Specificity')
            plt.tick_params(
                axis='x',          # changes apply to the x-axis
                which='both',      # both major and minor ticks are affected
                bottom='off',      # ticks along the bottom edge are off
                top='off',         # ticks along the top edge are off
                labelbottom='off')  # labels along the bottom edge are off
            plt.tight_layout()
            plt.show()

    return stats, predictions


def main():
    if len(sys.argv) == 1:
        print("TODO: Put in an example")
    elif len(sys.argv) != 3:
        raise IOError("This function accepts two arguments")
    else:
        prediction = sys.argv[1]
        patientinfo = sys.argv[2]
    plot_single_SVM(prediction, patientinfo)


if __name__ == '__main__':
    main()
