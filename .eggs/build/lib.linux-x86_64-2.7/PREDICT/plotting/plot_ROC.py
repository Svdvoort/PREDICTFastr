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

import matplotlib.pyplot as plt
import pandas as pd
import argparse

import numpy as np
from sklearn.metrics import roc_curve, auc
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

    # Loop over all svms and test sets
    # NOTE: sklearn advices decision_function for ROC, Sebastian uses predict?
    y_score = list()
    for num, X_test in enumerate(X_test_full):
        y_score.extend(svm[k].ix('svms')[0][num].decision_function(X_test))
        # y_score.extend(svm[k].ix('svms')[0][num].predict(X_test))

    # Convert y_test tuple to list
    y_test = list()
    for y in Y_test_full:
        y_test.extend(y)

    y_score = np.asarray(y_score)
    y_test = np.asarray(y_test)

    # Compute ROC curve and ROC area for each class
    fpr, tpr, _ = roc_curve(y_test, y_score)
    roc_auc = auc(fpr, tpr)

    # Compute micro-average ROC curve and ROC area
    f = plt.figure()
    subplot = f.add_subplot(111)
    lw = 2
    subplot.plot(fpr, tpr, color='darkorange',
             lw=lw, label='ROC curve (AUC = %0.2f)' % roc_auc)
    subplot.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (1 - Specificity)')
    plt.ylabel('True Positive Rate (Sensitivity)')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")
    # plt.show()

    output = args.svm.replace('.hdf5', '_roc.png')
    f.savefig(output)
    print(("ROC saved as {} !").format(output))

    # Save ROC values as JSON
    output = args.svm.replace('.hdf5', '_roc.json')
    with open(output, 'wb') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(['FPR', 'TPR'])
        for i in range(0, len(fpr)):
            data = [str(fpr[i]), str(tpr[i])]
            writer.writerow(data)

    print(("ROC saved as {} !").format(output))


if __name__ == '__main__':
    main()
