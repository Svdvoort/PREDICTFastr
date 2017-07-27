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
    y_test = list()
    y_predict = list()
    pid_test = list()
    for num, (X_test, yt, pidt) in enumerate(zip(X_test_full,
                                                 Y_test_full,
                                                 pid_test_full)):
        y_score.append(svm[k].ix('svms')[0][num].predict_proba(X_test))
        y_predict.append(svm[k].ix('svms')[0][num].predict(X_test))
        y_test.append(yt)
        pid_test.append(pidt)

    # Write output to csv
    output = args.svm.replace('.hdf5', '_allprobs.csv')
    with open(output, 'wb') as csv_file:
        writer = csv.writer(csv_file)

        for pid, ys, yt, yp in zip(pid_test, y_score, y_test, y_predict):
            writer.writerow(['', '', '', '', ''])
            writer.writerow(['Patient ID', 'Probability', 'Predicted Label', 'True Label', ''])

            for i in range(len(pid)):
                writer.writerow([str(pid[i]), str(ys[i][0]), str(yp[i]), str(yt[i])])


if __name__ == '__main__':
    main()
