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


import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
import sys
import compute_CI
import pandas as pd
import os
import PREDICT.genetics.genetic_processing as gp


def plot_SVM(prediction, label_data, label_type, show_plots=False,
             alpha=0.95, ensemble=False, verbose=True,
             ensemble_scoring=None, output='stats'):
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

    alpha: float, default 0.95
            Significance of confidence intervals.

    ensemble: False, integer or 'Caruana'
            Determine whether an ensemble will be created. If so,
            either provide an integer to determine how many of the
            top performing classifiers should be in the ensemble, or use
            the string "Caruana" to use smart ensembling based on
            Caruana et al. 2004.
    '''

    # Load the prediction object if it's a hdf5 file
    if type(prediction) is not pd.core.frame.DataFrame:
        if os.path.isfile(prediction):
            prediction = pd.read_hdf(prediction)

    # Select the estimator from the pandas dataframe to use
    keys = prediction.keys()
    SVMs = list()
    if label_type is None:
        label_type = keys[0]
    else:
        label_type = label_type

    # Extract the estimators, features and labels
    SVMs = prediction[label_type]['classifiers']
    Y_test = prediction[label_type]['Y_test']
    X_test = prediction[label_type]['X_test']
    X_train = prediction[label_type]['X_train']
    Y_train = prediction[label_type]['Y_train']
    feature_labels = prediction[label_type]['feature_labels']

    # Load the label data
    if type(label_data) is not dict:
        if os.path.isfile(label_data):
            label_data = gp.load_mutation_status(label_data, [[label_type]])

    patient_IDs = label_data['patient_IDs']
    mutation_label = label_data['mutation_label']

    # Create lists for performance measures
    sensitivity = list()
    specificity = list()
    precision = list()
    accuracy = list()
    auc = list()
    f1_score_list = list()
    patient_classification_list = dict()
    if output in ['scores', 'decision']:
        # Keep track of all groundth truths and scores
        y_truths = list()
        y_scores = list()
        y_predictions = list()
        PIDs = list()

    # Loop over the test sets, which probably correspond with cross validation
    # iterations
    for i in range(0, len(Y_test)):
        print("\n")
        print("Cross validation {} / {}.").format(str(i + 1), str(len(Y_test)))
        test_patient_IDs = prediction[label_type]['patient_ID_test'][i]
        train_patient_IDs = prediction[label_type]['patient_ID_train'][i]
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
            # NOTE: Added for backwards compatability
            if not hasattr(SVMs[i], 'cv_iter'):
                cv_iter = list(SVMs[i].cv.split(X_train_temp, Y_train_temp))
                SVMs[i].cv_iter = cv_iter

            # Create the ensemble
            X_train_temp = [(x, feature_labels) for x in X_train_temp]
            SVMs[i].create_ensemble(X_train_temp, Y_train_temp,
                                    method=ensemble, verbose=verbose,
                                    scoring=ensemble_scoring)

        # Create prediction
        y_prediction = SVMs[i].predict(X_test_temp)

        print("Truth: " + str(y_truth))
        print("Prediction: " + str(y_prediction))

        # Add if patient was classified correctly or not to counting
        for i_truth, i_predict, i_test_ID in zip(y_truth, y_prediction, test_patient_IDs):
            if i_truth == i_predict:
                patient_classification_list[i_test_ID]['N_correct'] += 1
            else:
                patient_classification_list[i_test_ID]['N_wrong'] += 1

        y_score = SVMs[i].decision_function(X_test_temp)
        print('AUC: ' + str(roc_auc_score(y_truth, y_score)))

        if output == 'decision':
            # Output the posteriors
            y_scores.append(y_score)
            y_truths.append(y_truth)
            y_predictions.append(y_prediction)
            PIDs.append(test_patient_IDs)

        elif output == 'scores':
            # Output the posteriors
            y_scores.append(SVMs[i].predict_proba(X_test_temp)[1])
            y_truths.append(y_truth)
            y_predictions.append(y_prediction)
            PIDs.append(test_patient_IDs)

        elif output == 'stats':
            # Compute statistics
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

    if output in ['scores', 'decision']:
        # Return the scores and true values of all patients
        return y_truths, y_scores, y_predictions, PIDs
    elif output == 'stats':
        # Compute statistics
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
            percentages[i_ID] = str(label) + ': ' + str(round(percentage_right, 2) * 100) + '%'
            if percentage_right == 1.0:
                alwaysright[i_ID] = label
                print(("Always Right: {}, label {}").format(i_ID, label))

            elif percentage_right == 0:
                alwayswrong[i_ID] = label
                print(("Always Wrong: {}, label {}").format(i_ID, label))

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


def main():
    if len(sys.argv) == 1:
        print("TODO: Put in an example")
    elif len(sys.argv) != 4:
        raise IOError("This function accepts three arguments!")
    else:
        prediction = sys.argv[1]
        patientinfo = sys.argv[2]
        label_type = sys.argv[3]
    plot_SVM(prediction, patientinfo, label_type)


if __name__ == '__main__':
    main()
