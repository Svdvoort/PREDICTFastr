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
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score


def main():
    parser = argparse.ArgumentParser(description='Radiomics results')
    parser.add_argument('-svm', '--svm', metavar='svm',
                        nargs='+', dest='svm', type=str, required=True,
                        help='SVM file (HDF)')
    args = parser.parse_args()

    if type(args.svm) is list:
        args.svm = ''.join(args.svm)

    svms = pd.read_hdf(args.svm)

    svms = svms[svms.keys()[0]]

    sensitivity = list()
    specificity = list()
    auc = list()
    for svm, X_test, Y_test in zip(svms.svms, svms.X_test, svms.Y_test):
        Y_prediction = svm.predict(X_test)
        print Y_prediction
        c_mat = confusion_matrix(Y_test, Y_prediction)
        TN = c_mat[0, 0]
        FN = c_mat[1, 0]
        TP = c_mat[1, 1]
        FP = c_mat[0, 1]
        print confusion_matrix
        raise IOError

        if FN == 0 and TP == 0:
            sensitivity.append(0)
        else:
            sensitivity.append(float(TP)/(TP+FN))

        if FP == 0 and TN == 0:
            specificity.append(0)
        else:
            specificity.append(float(TN)/(FP+TN))

        auc.append(roc_auc_score(Y_test, Y_prediction))

    for idx, _ in enumerate(sensitivity):
        print auc[idx], sensitivity[idx], specificity[idx]

    print np.mean(auc), np.mean(sensitivity), np.mean(specificity)


if __name__ == '__main__':
    main()
