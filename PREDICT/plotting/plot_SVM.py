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
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
import sys
import compute_CI
import pandas as pd
import os

import PREDICT.genetics.genetic_processing as gp


def plot_single_SVM(prediction, mutation_data, label_type, show_plots=False):
    if type(prediction) is not pd.core.frame.DataFrame:
        if os.path.isfile(prediction):
            prediction = pd.read_hdf(prediction)

    keys = prediction.keys()
    SVMs = list()
    label = keys[0]
    SVMs = prediction[label]['classifiers']

    Y_test = prediction[label]['Y_test']
    X_test = prediction[label]['X_test']
    Y_train = prediction[label]['X_train']

    # print(len(X_test[0][0]))
    # print(config)
    # X_train = data2['19q']['X_train']
    # Y_train = data2['19q']['Y_train']
    # mutation_data = gp.load_mutation_status(patientinfo, [[label]])
    if type(mutation_data) is not dict:
        if os.path.isfile(mutation_data):
            mutation_data = gp.load_mutation_status(mutation_data, [[label_type]])

    patient_IDs = mutation_data['patient_IDs']
    mutation_label = mutation_data['mutation_label']
    # mutation_name = mutation_data['mutation_name']

    # print(len(SVMs))
    N_iterations = float(len(SVMs))

    # mutation_label = np.asarray(mutation_label)

    sensitivity = list()
    specificity = list()
    precision = list()
    accuracy = list()
    auc = list()
    # auc_train = list()
    f1_score_list = list()

    patient_classification_list = dict()

    for i in range(0, len(Y_test)):
        # print(Y_test[i])
        # if Y_test[i].shape[1] > 1:
        #     # print(Y_test[i])
        #     y_truth = np.prod(Y_test[i][:, 0:2], axis=1)
        # else:
        #     y_truth_test = Y_test[i]
        test_patient_IDs = prediction[label]['patient_ID_test'][i]

        if 'LGG-Radiogenomics-046' in test_patient_IDs:
            wrong_index = np.where(test_patient_IDs == 'LGG-Radiogenomics-046')
            test_patient_IDs = np.delete(test_patient_IDs, wrong_index)
            X_temp = X_test[i]
            print(X_temp.shape)
            X_temp = np.delete(X_test[i], wrong_index, axis=0)
            print(X_temp.shape)

            # X_test.pop(wrong_index[0])

            # print(len(X_test))
        else:
            X_temp = X_test[i]

        test_indices = list()
        for i_ID in test_patient_IDs:
            test_indices.append(np.where(patient_IDs == i_ID)[0][0])

            if i_ID not in patient_classification_list:
                patient_classification_list[i_ID] = dict()
                patient_classification_list[i_ID]['N_test'] = 0
                patient_classification_list[i_ID]['N_correct'] = 0
                patient_classification_list[i_ID]['N_wrong'] = 0

            patient_classification_list[i_ID]['N_test'] += 1

        y_truth = [mutation_label[0][k] for k in test_indices]
        # print(y_truth)
        # print(y_truth_test)
        # print(test_patient_IDs)

        y_predict_1 = SVMs[i].predict(X_temp)

        # print(y_predict_1).shape

        y_prediction = y_predict_1
        # y_prediction = np.prod(y_prediction, axis=0)

        print "Truth: ", y_truth
        print "Prediction: ", y_prediction

        for i_truth, i_predict, i_test_ID in zip(y_truth, y_prediction, test_patient_IDs):
            if i_truth == i_predict:
                patient_classification_list[i_test_ID]['N_correct'] += 1
            else:
                patient_classification_list[i_test_ID]['N_wrong'] += 1

        # print('bla')
        # print(y_truth)
        # print(y_prediction)

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
        accuracy.append(accuracy_score(y_truth, y_prediction))
        y_score = SVMs[i].decision_function(X_temp)
        auc.append(roc_auc_score(y_truth, y_score))
        f1_score_list.append(f1_score(y_truth, y_prediction, average='weighted'))

    # Adjusted according to "Inference for the Generelization error"

    accuracy_mean = np.mean(accuracy)
    S_uj = 1.0 / max((N_iterations - 1), 1) * np.sum((accuracy_mean - accuracy)**2.0)

    print Y_test
    N_1 = float(len(Y_train[0]))
    N_2 = float(len(Y_test[0]))

    print(N_1)
    print(N_2)

    accuracy_var = np.sqrt((1.0/N_iterations + N_2/N_1)*S_uj)
    print(accuracy_var)
    print(np.sqrt(1/N_iterations*S_uj))
    print(st.sem(accuracy))

    stats = dict()
    stats["Accuracy 95%:"] = str(compute_CI.compute_confidence(accuracy, N_1, N_2, 0.95))

    stats["AUC 95%:"] = str(compute_CI.compute_confidence(auc, N_1, N_2, 0.95))

    stats["F1-score 95%:"] = str(compute_CI.compute_confidence(f1_score_list, N_1, N_2, 0.95))

    stats["Precision 95%:"] = str(compute_CI.compute_confidence(precision, N_1, N_2, 0.95))

    stats["Sensitivity 95%: "] = str(compute_CI.compute_confidence(sensitivity, N_1, N_2, 0.95))

    stats["Specificity 95%:"] = str(compute_CI.compute_confidence(specificity, N_1, N_2, 0.95))

    print("Accuracy 95%:" + str(compute_CI.compute_confidence(accuracy, N_1, N_2, 0.95)))

    print("AUC 95%:" + str(compute_CI.compute_confidence(auc, N_1, N_2, 0.95)))

    print("F1-score 95%:" + str(compute_CI.compute_confidence(f1_score_list, N_1, N_2, 0.95)))

    print("Precision 95%:" + str(compute_CI.compute_confidence(precision, N_1, N_2, 0.95)))

    print("Sensitivity 95%: " + str(compute_CI.compute_confidence(sensitivity, N_1, N_2, 0.95)))

    print("Specificity 95%:" + str(compute_CI.compute_confidence(specificity, N_1, N_2, 0.95)))

    alwaysright = dict()
    alwayswrong = dict()
    for i_ID in patient_classification_list:
        percentage_right = patient_classification_list[i_ID]['N_correct'] / float(patient_classification_list[i_ID]['N_test'])

        # print(i_ID + ' , ' + str(patient_classification_list[i_ID]['N_test']) + ' : ' + str(percentage_right) + '\n')
        if percentage_right == 1.0:
            label = mutation_label[0][np.where(i_ID == patient_IDs)]
            label = label[0][0]
            alwaysright[i_ID] = label
            # alwaysright.append(('{} ({})').format(i_ID, label))
            print(("Always Right: {}, label {}").format(i_ID, label))

        if percentage_right == 0:
            label = mutation_label[0][np.where(i_ID == patient_IDs)].tolist()
            label = label[0][0]
            alwayswrong[i_ID] = label
            # alwayswrong.append(('{} ({})').format(i_ID, label))
            print(("Always Wrong: {}, label {}").format(i_ID, label))

    stats["Always right"] = alwaysright
    stats["Always wrong"] = alwayswrong

    if show_plots:
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
    elif len(sys.argv) != 3:
        raise IOError("This function accepts two arguments")
    else:
        prediction = sys.argv[1]
        patientinfo = sys.argv[2]
    plot_single_SVM(prediction, patientinfo)


if __name__ == '__main__':
    main()
