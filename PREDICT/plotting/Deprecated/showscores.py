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

import pandas as pd
import argparse
import numpy as np
import csv


def main():
    parser = argparse.ArgumentParser(description='Radiomics results')
    parser.add_argument('-svm', '--svm', metavar='svm',
                        nargs='+', dest='svm', type=str, required=True,
                        help='SVM file (HDF)')
    args = parser.parse_args()

    if type(args.svm) is list:
        args.svm = ''.join(args.svm)

    svm = pd.read_hdf(args.svm)

    k = svm.keys()[0]
    X_test_full = svm[k].X_test
    Y_test_full = svm[k].Y_test
    pid_test_full = svm[k].patient_ID_test

    # Loop over all svms and test sets
    # NOTE: sklearn advices decision_function for ROC, Sebastian uses predict?
    y_score = list()
    y_predict = list()
    y_test = list()
    pid_test = list()
    for num, (X_test, yt, pidt) in enumerate(zip(X_test_full,
                                                 Y_test_full,
                                                 pid_test_full)):
        y_score.extend(svm[k].ix('svms')[0][num].predict_proba(X_test))
        y_predict.extend(svm[k].ix('svms')[0][num].predict(X_test))
        y_test.extend(yt)
        pid_test.extend(pidt)

    # Gather all scores for all patients and average
    wrongs = dict()
    rights = dict()
    pid_unique = list(set(pid_test))
    pid_unique = sorted(pid_unique)
    for pid in pid_unique:
        class1_scores = list()
        class2_scores = list()
        labels = list()
        for num, allid in enumerate(pid_test):
            if allid == pid:
                class1_scores.append(y_score[num][0])
                class2_scores.append(y_score[num][1])
                labels.append(y_predict[num])
                truelabel = y_test[num]

        right_probs = list()
        wrong_probs = list()
        for num, label in enumerate(labels):
            if label == truelabel:
                right_probs.append(class1_scores[num])
            else:
                wrong_probs.append(class1_scores[num])

        if len(wrong_probs) > len(right_probs):
            wrongs[pid] = [np.mean(wrong_probs), truelabel]
        else:
            rights[pid] = [np.mean(right_probs), truelabel]

    # Sort the keys based on the probabilities
    WK = wrongs.keys()
    WV = list()
    for sk in WK:
        WV.append(wrongs[sk][0])

    RK = rights.keys()
    RV = list()
    for sk in RK:
        RV.append(rights[sk][0])

    wk = [wk for (wv, wk) in sorted(zip(WV, WK))]
    rk = [rk for (rv, rk) in sorted(zip(RV, RK))]
    wk = list(reversed(wk))
    rk = list(reversed(rk))

    # Write output to csv
    output = args.svm.replace('.hdf5', '_scores.csv')
    maxlen = max(len(wk), len(rk))
    with open(output, 'wb') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(['Rights', '', '', '', 'Wrongs', '', ''])
        writer.writerow(['Patient ID', 'Probability', 'True Label', '',
                         'Patient ID', 'Probability', 'True Label'])
        for i in range(0, maxlen):
            data = ['', '', '', '', '', '', '']
            if i < len(rk):
                data[0:3] = [rk[i], rights[rk[i]][0], rights[rk[i]][1]]

            if i < len(wk):
                data[4:6] = [wk[i], wrongs[wk[i]][0], wrongs[wk[i]][1]]

            writer.writerow(data)

    # Loop over all svms and test sets
    # NOTE: sklearn advices decision_function for ROC, Sebastian uses predict?
    y_score = list()
    y_test = list()
    pid_test = list()
    for num, (X_test, yt, pidt) in enumerate(zip(X_test_full,
                                                 Y_test_full,
                                                 pid_test_full)):
        y_score.extend(svm[k].ix('svms')[0][num].decision_function(X_test))
        y_test.extend(yt)
        pid_test.extend(pidt)

    # Gather all scores for all patients and average
    wrongs = dict()
    rights = dict()
    pid_unique = list(set(pid_test))
    pid_unique = sorted(pid_unique)
    meanscores = list()
    for pid in pid_unique:
        scores = list()
        for num, allid in enumerate(pid_test):
            if allid == pid:
                scores.append(y_score[num])
                truelabel = y_test[num]

        meanscore = np.mean(scores)
        if meanscore > 0:
            predlabel = 1
        else:
            predlabel = 0

        if truelabel != predlabel:
            wrongs[pid] = [meanscore, truelabel]
        else:
            rights[pid] = [meanscore, truelabel]

        meanscores.append(meanscore)

    # We let the mean score for each class be -1/1
    meanscore = np.mean(np.abs(meanscores))

    # Sort the keys based on the distances
    WK = wrongs.keys()
    WV = list()
    for sk in WK:
        WV.append(wrongs[sk][0])

    RK = rights.keys()
    RV = list()
    for sk in RK:
        RV.append(rights[sk][0])

    wk = [wk for (wv, wk) in sorted(zip(np.abs(WV), WK))]
    rk = [rk for (rv, rk) in sorted(zip(np.abs(RV), RK))]
    wk = list(reversed(wk))
    rk = list(reversed(rk))

    # Write output to csv
    output = args.svm.replace('.hdf5', '_distance.csv')
    maxlen = max(len(wk), len(rk))
    with open(output, 'wb') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(['Rights', '', '', '', 'Wrongs', '', ''])
        writer.writerow(['Patient ID', 'Distance', 'True Label', '',
                         'Patient ID', 'Distance', 'True Label'])
        for i in range(0, maxlen):
            data = ['', '', '', '', '', '', '']

            if i < len(rk):
                data[0:3] = [rk[i], rights[rk[i]][0]/meanscore, rights[rk[i]][1]]

            if i < len(wk):
                data[4:6] = [wk[i], wrongs[wk[i]][0]/meanscore, wrongs[wk[i]][1]]

            writer.writerow(data)

if __name__ == '__main__':
    main()
