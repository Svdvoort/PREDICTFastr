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
import os
import glob


def main():
    parser = argparse.ArgumentParser(description='Radiomics results')
    parser.add_argument('-svm', '--svm', metavar='svm',
                        nargs='+', dest='svm', type=str, required=True,
                        help='SVM file (HDF)')
    args = parser.parse_args()

    if type(args.svm) is list:
        args.svm = ''.join(args.svm)

    if os.path.isdir(args.svm):
        posteriors = readmultiplesvm(args.svm)
        output = os.path.join(args.svm, 'posteriors.csv')
    else:
        posteriors = readsinglesvm(args.svm)
        output = args.svm.replace('.hdf5', '_posteriors.csv')

    # Write output to csv
    with open(output, 'wb') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(['Patient ID', 'Posterior', 'True Label', 'Counts'])
        for k in posteriors.keys():
            p = posteriors[k]
            data = [k, str(p[0]), str(p[1]), str(p[2])]

            writer.writerow(data)


def readmultiplesvm(svmdir):
    # Search the directory for all svm hdf5s
    svms = glob.glob(svmdir + '/*.hdf5')

    # Compute test set scores for each svm
    y_score = list()
    y_test = list()
    pid_test = list()
    y_predict = list()
    for num, svm in enumerate(svms):
        print("Processing svm {} / {}.").format(str(num+1), len(svms))
        svm = pd.read_hdf(svm)

        k = svm.keys()[0]
        X_test = svm[k].X_test
        Y_test = svm[k].Y_test
        pidt = svm[k].patient_ID_test

        y_score.extend(svm[k].ix('svms')[0].predict_proba(X_test))
        y_predict.extend(svm[k].ix('svms')[0].predict(X_test))
        y_test.extend(Y_test)
        pid_test.extend(pidt)

    # Gather all scores for all patients and average
    pid_unique = list(set(pid_test))
    pid_unique = sorted(pid_unique)
    posteriors = dict()
    for pid in pid_unique:
        posteriors[pid] = list()

        counts = 0
        for num, allid in enumerate(pid_test):
            if allid == pid:
                counts += 1
                posteriors[pid].append(y_score[num][0])
                truelabel = y_test[num]

        posteriors[pid] = [np.mean(posteriors[pid]), truelabel, counts]

    return posteriors


def readsinglesvm(svmfile):
    svm = pd.read_hdf(svmfile)

    k = svm.keys()[0]
    X_test_full = svm[k].X_test
    Y_test_full = svm[k].Y_test
    pid_test_full = svm[k].patient_ID_test

    # Loop over all svms and test sets
    # NOTE: sklearn advices decision_function for ROC, Sebastian uses predict?
    y_score = list()
    y_test = list()
    pid_test = list()
    y_predict = list()
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
    posteriors = dict()
    for pid in pid_unique:
        posteriors[pid] = list()
        wrongs[pid] = list()
        rights[pid] = list()

        counts = 0
        for num, allid in enumerate(pid_test):
            if allid == pid:
                counts += 1
                posteriors[pid].append(y_score[num][0])
                # predlabel = y_predict[num]
                truelabel = y_test[num]

            # if truelabel != predlabel:
                # wrongs[pid].append([posterior, truelabel])
            # else:
                # rights[pid].append([posterior, truelabel])

        posteriors[pid] = [np.mean(posteriors[pid]), truelabel, counts]

    return posteriors


if __name__ == '__main__':
    main()
