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

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Ellipse
    from matplotlib2tikz import save as tikz_save
except ImportError:
    print("[PREDICT Warning] Cannot use plot_ROC function, as _tkinter is not installed")

import pandas as pd
import argparse
from compute_CI import compute_confidence_logit as CIl
from compute_CI import compute_confidence as CI
import numpy as np
from sklearn.metrics import roc_curve, auc
# from sklearn.metrics import roc_auc_score
import csv
import glob
import natsort
import os
import json

from PREDICT.processing.fitandscore import fit_and_score
from sklearn.base import clone



def new_ROC(L, f, verbose=False):
    # fpr_old, tpr_old, _ = roc_curve(L, f)
    # Sort both lists based on scores
    # f -= np.min(f)
    # f = f / np.max(np.abs(f))

    # Added
    L = [int(l) for l in L]

    L = np.asarray(L)
    f = np.asarray(f)
    inds = f.argsort()
    Lsorted = L[inds]
    f = f[inds]

    # Compute the TPR and FPR
    FP = 0
    TP = 0
    fpr = list()
    tpr = list()
    thresholds = list()
    fprev = -np.inf
    i = 0
    N = float(np.bincount(L)[0])
    P = float(np.bincount(L)[1])

    while i < len(Lsorted):
        if f[i] != fprev:
            fpr.append(1 - FP/N)
            tpr.append(1 - TP/P)
            thresholds.append(f[i])
            fprev = f[i]

        if Lsorted[i] == 1:
            TP += 1
        else:
            FP += 1

        i += 1

    if verbose:
        print fpr, tpr
        roc_auc = auc(fpr, tpr)
        plt.figure()
        lw = 2
        plt.plot(fpr, tpr, color='darkorange',
                 lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
        plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic example')
        plt.legend(loc="lower right")

        # roc_auc = auc(fpr_old, tpr_old)
        # plt.figure()
        # lw = 2
        # plt.plot(fpr_old, tpr_old, color='darkorange',
        #          lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
        # plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        # plt.xlim([0.0, 1.0])
        # plt.ylim([0.0, 1.05])
        # plt.xlabel('False Positive Rate')
        # plt.ylabel('True Positive Rate')
        # plt.title('Receiver operating characteristic example')
        # plt.legend(loc="lower right")
        # plt.show()

    return fpr[::-1], tpr[::-1], thresholds[::-1]


def ROC_thresholding(fprt, tprt, thresholds, nsamples=21):
    T = list()
    for t in thresholds:
        T.extend(t)
    T = sorted(T)
    nrocs = len(fprt)
    fpr = np.zeros((nsamples, nrocs))
    tpr = np.zeros((nsamples, nrocs))
    tsamples = np.linspace(0, len(T) - 1, nsamples)
    th = list()
    for n, tidx in enumerate(tsamples):
        tidx = int(tidx)
        th.append(T[tidx])
        for i in range(0, nrocs):
            idx = 0
            while float(thresholds[i][idx]) > float(T[tidx]) and idx < (len(thresholds[i]) - 1):
                idx += 1
            fpr[n, i] = fprt[i][idx]
            tpr[n, i] = tprt[i][idx]

    return fpr, tpr, th


def ROC_averaging(fprt, tprt, nsamples=21):
    # Compute ROC curve and ROC area for each class
    fpr = list()
    tpr = list()
    fpr_plot = np.linspace(0, 1, nsamples)
    for fpr_temp, tpr_temp in zip(fprt, tprt):
        tpr_temp = np.interp(x=fpr_plot, xp=fpr_temp, fp=tpr_temp)

        if type(tpr) is list:
            tpr = tpr_temp
        else:
            tpr = np.column_stack((tpr, tpr_temp))

        if type(fpr) is list:
            fpr = fpr_plot
        else:
            fpr = np.column_stack((fpr, fpr_plot))

    return fpr, tpr


def plot_ROC_CIl(y_test, y_score, N_1, N_2, plot='default', alpha=0.95, verbose=True):
    # Compute ROC curve and ROC area for each class
    fpr = list()
    tpr = list()
    roc_auc = list()
    fpr_plot = np.linspace(0, 1, 11)
    # auc_min = 1
    # auc_max = 0
    for yt, ys in zip(y_test, y_score):
        fpr_temp, tpr_temp, _ = roc_curve(yt, ys)

        # Need to resample, as we need the same lengt for all fpr and tpr
        tpr_temp = np.interp(x=fpr_plot, xp=fpr_temp, fp=tpr_temp)

        # Replace the zeros and ones for the logit
        tpr_temp = np.asarray([0.001 if x == 0 else x for x in tpr_temp])
        tpr_temp = np.asarray([0.999 if x == 1 else x for x in tpr_temp])
        auc_temp = auc(fpr_plot, tpr_temp)
        roc_auc.append(auc_temp)
        # if auc_temp > auc_max:
        #     auc_max = auc_temp
        #     tpr_max = tpr_temp
        # if auc_temp < auc_min:
        #     auc_min = auc_temp
        #     tpr_min = tpr_temp

        if type(tpr) is list:
            tpr = tpr_temp
        else:
            tpr = np.column_stack((tpr, tpr_temp))

        if type(fpr) is list:
            fpr = fpr_plot
        else:
            fpr = np.column_stack((fpr, fpr_plot))

    CIs_tpr = list()
    CIs_tpr_nl = list()
    means_tpr = list()
    for i in range(0, len(fpr_plot)):
        cit = CIl(tpr[i, :], N_1, N_2, alpha)
        citnl = CI(tpr[i, :], N_1, N_2, alpha)
        CIs_tpr_nl.append([citnl[0], citnl[1]])
        CIs_tpr.append([cit[0], cit[1]])
        # means_tpr.append(mean)
    CIs_tpr = np.asarray(CIs_tpr)
    CIs_tpr_nl = np.asarray(CIs_tpr_nl)
    means_tpr = np.mean(CIs_tpr, axis=1)

    # compute AUC CI
    roc_auc2 = CI(roc_auc, N_1, N_2, alpha)
    roc_auc = CIl(roc_auc, N_1, N_2, alpha)

    print roc_auc

    # Visualize with coloring between lines
    # if plot == 'default':
    tpr_plot = means_tpr
    tpr_plot2 = np.mean(CIs_tpr_nl, axis=1)
    # elif plot == 'max':
    #     tpr_plot = tpr_max
    # elif plot == 'min':
    #     tpr_plot = tpr_min

    f = plt.figure()
    lw = 2
    subplot = f.add_subplot(111)
    subplot.plot(fpr_plot, tpr_plot, color='darkorange',
                 lw=lw, label='ROC curve (AUC = (%0.2f, %0.2f))' % (roc_auc[0], roc_auc[1]))
    subplot.fill_between(fpr_plot, CIs_tpr[:, 0], CIs_tpr[:, 1], facecolor='darkorange', alpha=0.5)
    subplot.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (1 - Specificity)')
    plt.ylabel('True Positive Rate (Sensitivity)')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")

    f = plt.figure()
    lw = 2
    subplot = f.add_subplot(111)
    subplot.plot(fpr_plot, tpr_plot2, color='darkorange',
                 lw=lw, label='ROC curve (AUC = (%0.2f, %0.2f))' % (roc_auc2[0], roc_auc2[1]))
    subplot.fill_between(fpr_plot, CIs_tpr_nl[:, 0], CIs_tpr_nl[:, 1], facecolor='darkorange', alpha=0.5)
    subplot.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (1 - Specificity)')
    plt.ylabel('True Positive Rate (Sensitivity)')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")

    if verbose:
        plt.show()
        print fpr_plot, tpr_plot, CIs_tpr[:, 0], CIs_tpr[:, 1]

        # Need to plot again, since we save it after closing
        f = plt.figure()
        lw = 2
        subplot = f.add_subplot(111)
        subplot.plot(fpr_plot, tpr_plot, color='darkorange',
                     lw=lw, label='ROC curve (AUC = (%0.2f, %0.2f))' % (roc_auc[0], roc_auc[1]))
        subplot.fill_between(fpr_plot, CIs_tpr[:, 0], CIs_tpr[:, 1], facecolor='darkorange', alpha=0.5)
        subplot.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate (1 - Specificity)')
        plt.ylabel('True Positive Rate (Sensitivity)')
        plt.title('Receiver operating characteristic')
        plt.legend(loc="lower right")

    return f, fpr, tpr


def plot_ROC_CIc(y_test, y_score, N_1, N_2, plot='default', alpha=0.95, verbose=True):
    # Compute ROC curve and ROC area for each class
    fprt = list()
    tprt = list()
    roc_auc = list()
    thresholds = list()
    for yt, ys in zip(y_test, y_score):
        fpr_temp, tpr_temp, thresholds_temp = new_ROC(yt, ys)
        # f = plt.figure()
        # subplot = f.add_subplot(111)
        # lw = 2
        # subplot.plot(fpr_temp, tpr_temp, color='darkorange',
        #              lw=lw, label='ROC curve (AUC = %0.2f)' % auc(fpr_temp, tpr_temp))
        # subplot.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        # plt.xlim([0.0, 1.0])
        # plt.ylim([0.0, 1.05])
        # plt.xlabel('False Positive Rate (1 - Specificity)')
        # plt.ylabel('True Positive Rate (Sensitivity)')
        # plt.title('Receiver operating characteristic')
        # plt.legend(loc="lower right")
        # plt.show()
        roc_auc.append(auc(fpr_temp, tpr_temp))
        fprt.append(fpr_temp)
        tprt.append(tpr_temp)
        thresholds.append(thresholds_temp)

    tsamples = 21
    fpr, tpr, th = ROC_thresholding(fprt, tprt, thresholds, tsamples)
    # fpr, tpr = ROC_averaging(fprt, tprt, tsamples)

    CIs_tpr = list()
    CIs_fpr = list()
    for i in range(0, tsamples):
        cit_fpr = CI(fpr[i, :], N_1, N_2, alpha)
        CIs_fpr.append([cit_fpr[0], cit_fpr[1]])
        cit_tpr = CI(tpr[i, :], N_1, N_2, alpha)
        CIs_tpr.append([cit_tpr[0], cit_tpr[1]])

    CIs_tpr = np.asarray(CIs_tpr)
    CIs_fpr = np.asarray(CIs_fpr)
    CIs_tpr_means = np.mean(CIs_tpr, axis=1)
    CIs_fpr_means = np.mean(CIs_fpr, axis=1)

    # compute AUC CI
    roc_auc = CI(roc_auc, N_1, N_2, alpha)

    f = plt.figure()
    lw = 2
    subplot = f.add_subplot(111)
    subplot.plot(CIs_fpr_means, CIs_tpr_means, color='black',
                 lw=lw, label='ROC curve (AUC = (%0.2f, %0.2f))' % (roc_auc[0], roc_auc[1]))

    for i in range(0, len(CIs_fpr_means)):
        if CIs_tpr[i, 1] <= 1:
            ymax = CIs_tpr[i, 1]
        else:
            ymax = 1

        if CIs_tpr[i, 0] <= 0:
            ymin = 0
        else:
            ymin = CIs_tpr[i, 0]

        if CIs_tpr_means[i] <= 1:
            ymean = CIs_tpr_means[i]
        else:
            ymean = 1

        if CIs_fpr[i, 1] <= 1:
            xmax = CIs_fpr[i, 1]
        else:
            xmax = 1

        if CIs_fpr[i, 0] <= 0:
            xmin = 0
        else:
            xmin = CIs_fpr[i, 0]

        if CIs_fpr_means[i] <= 1:
            xmean = CIs_fpr_means[i]
        else:
            xmean = 1

        subplot.plot([xmin, xmax],
                     [ymean, ymean],
                     color='black', alpha=0.15)
        subplot.plot([xmean, xmean],
                     [ymin, ymax],
                     color='black', alpha=0.15)

        # e = Ellipse(xy=[CIs_fpr_means[i], CIs_tpr_means[i]],
        #             height=(CIs_tpr[i, 1] - CIs_tpr[i, 0])/dw2,
        #             width=(CIs_fpr[i, 1] - CIs_fpr[i, 0])/dw2)
        # subplot.add_artist(e)
        # e.set_clip_box(subplot.bbox)
        # e.set_alpha(1)
        # e.set_facecolor([1, 0.75, 0.25])

    subplot.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (1 - Specificity)')
    plt.ylabel('True Positive Rate (Sensitivity)')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")

    if verbose:
        plt.show()
        print CIs_fpr_means, CIs_tpr_means, CIs_tpr[:, 0], CIs_tpr[:, 1], CIs_fpr[:, 0], CIs_fpr[:, 1]

        f = plt.figure()
        lw = 2
        subplot = f.add_subplot(111)
        subplot.plot(CIs_fpr_means, CIs_tpr_means, color='darkorange',
                     lw=lw, label='ROC curve (AUC = (%0.2f, %0.2f))' % (roc_auc[0], roc_auc[1]))

        for i in range(0, len(CIs_fpr_means)):
            if CIs_tpr[i, 1] <= 1:
                ymax = CIs_tpr[i, 1]
            else:
                ymax = 1

            if CIs_tpr[i, 0] <= 0:
                ymin = 0
            else:
                ymin = CIs_tpr[i, 0]

            if CIs_tpr_means[i] <= 1:
                ymean = CIs_tpr_means[i]
            else:
                ymean = 1

            if CIs_fpr[i, 1] <= 1:
                xmax = CIs_fpr[i, 1]
            else:
                xmax = 1

            if CIs_fpr[i, 0] <= 0:
                xmin = 0
            else:
                xmin = CIs_fpr[i, 0]

            if CIs_fpr_means[i] <= 1:
                xmean = CIs_fpr_means[i]
            else:
                xmean = 1

            subplot.plot([xmin, xmax],
                         [ymean, ymean],
                         color='black', alpha=0.15)
            subplot.plot([xmean, xmean],
                         [ymin, ymax],
                         color='black', alpha=0.15)

        subplot.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate (1 - Specificity)')
        plt.ylabel('True Positive Rate (Sensitivity)')
        plt.title('Receiver operating characteristic')
        plt.legend(loc="lower right")

    return f, CIs_fpr, CIs_tpr


def plot_ROC(y_test, y_score, N_1, N_2, alpha=0.95, verbose=True):
    # Compute ROC curve and ROC area for each class
    fpr = list()
    tpr = list()
    roc_auc = list()
    fpr_plot = np.linspace(0, 1, 11)
    for yt, ys in zip(y_test, y_score):
        fpr_temp, tpr_temp, _ = roc_curve(yt, ys)

        # Need to resample, as we need the same lengt for all fpr and tpr
        tpr_temp = np.interp(x=fpr_plot, xp=fpr_temp, fp=tpr_temp)

        # Replace the zeros and ones for the logit
        auc_temp = auc(fpr_plot, tpr_temp)
        roc_auc.append(auc_temp)

        # Stack in arrays so we can compute the CIs later
        if type(tpr) is list:
            tpr = tpr_temp
        else:
            tpr = np.column_stack((tpr, tpr_temp))

        if type(fpr) is list:
            fpr = fpr_plot
        else:
            fpr = np.column_stack((fpr, fpr_plot))

    # Compute CIs
    CIs_tpr = list()
    for i in range(0, len(fpr_plot)):
        cit = CI(tpr[i, :], N_1, N_2, alpha)
        CIs_tpr.append([cit[0], cit[1]])
    CIs_tpr = np.asarray(CIs_tpr)
    means_CI = np.mean(CIs_tpr, axis=1)
    print np.mean(CIs_tpr, axis=1).shape
    print np.mean(CIs_tpr, axis=0).shape
    print np.mean(CIs_tpr).shape

    # compute AUC CI
    roc_auc = CI(roc_auc, N_1, N_2, alpha)

    # Compute micro-average ROC curve and ROC area
    f = plt.figure()
    subplot = f.add_subplot(111)
    lw = 2
    subplot.plot(fpr_plot, means_CI, color='darkorange',
                 lw=lw, label='ROC curve (AUC = (%0.2f, %0.2f))' % (roc_auc[0], roc_auc[1]))
    subplot.fill_between(fpr_plot, CIs_tpr[:, 0], CIs_tpr[:, 1], facecolor='darkorange', alpha=0.5)
    subplot.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (1 - Specificity)')
    plt.ylabel('True Positive Rate (Sensitivity)')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")
    if verbose:
        plt.show()

        f = plt.figure()
        subplot = f.add_subplot(111)
        lw = 2
        subplot.plot(fpr_plot, means_CI, color='darkorange',
                     lw=lw, label='ROC curve (AUC = (%0.2f, %0.2f))' % (roc_auc[0], roc_auc[1]))
        subplot.fill_between(fpr_plot, CIs_tpr[:, 0], CIs_tpr[:, 1], facecolor='darkorange', alpha=0.5)
        subplot.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate (1 - Specificity)')
        plt.ylabel('True Positive Rate (Sensitivity)')
        plt.title('Receiver operating characteristic')
        plt.legend(loc="lower right")

    return f, fpr, tpr


def main_old():
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
    Y_train_full = svm[k].Y_train

    # Loop over all svms and test sets
    # NOTE: sklearn advices decision_function for ROC, Sebastian uses predict?
    y_score = list()
    for num, X_test in enumerate(X_test_full):
        y_score.append(svm[k].ix('svms')[0][num].decision_function(X_test))
        # y_score.extend(svm[k].ix('svms')[0][num].predict(X_test))

    # Convert y_test tuple to list
    y_test = list()
    for y in Y_test_full:
        y_test.append(y)

    # y_score = np.asarray(y_score)
    # y_test = np.asarray(y_test)

    N_1 = float(len(svm[k].patient_ID_train[0]))
    N_2 = float(len(svm[k].patient_ID_test[0]))

    f, fpr, tpr = plot_ROC_CI(y_test, y_score, N_1, N_2)
    raise IOError

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


def main_1class():
    parser = argparse.ArgumentParser(description='Radiomics results')
    parser.add_argument('-svm', '--svm', metavar='svm',
                        nargs='+', dest='svm', type=str, required=True,
                        help='SVM file (HDF)')
    args = parser.parse_args()

    if type(args.svm) is list:
        args.svm = ''.join(args.svm)

    if os.path.isfile(args.svm):
        with open(args.svm, 'r') as fp:
            scores = json.load(fp)

        y_test = scores['y_test']
        y_score = scores['y_score']
        N_1 = scores['N_1']
        N_2 = scores['N_2']

        args.svm = os.path.dirname(args.svm)

    else:
        svms = glob.glob(args.svm + '/*.hdf5')

        # Loop over all svms and test sets
        # NOTE: sklearn advices decision_function for ROC, Sebastian uses predict?
        y_score = list()
        y_test = list()
        svms = natsort.natsorted(svms)
        # svms = svms[:10]
        for num, svm in enumerate(svms):
            print("Processing svm {} / {}.").format(str(num+1), str(len(svms)))
            svm = pd.read_hdf(svm)
            k = svm.keys()[0]
            X_test = svm[k].X_test
            Y_test = svm[k].Y_test
            y_score.append(svm[k].trained_classifier.decision_function(X_test).tolist())
            y_test.append(Y_test.tolist())

        N_1 = float(len(svm[k].patient_ID_train[0]))
        N_2 = float(len(svm[k].patient_ID_test[0]))

        scores = dict()
        scores['y_test'] = y_test
        scores['y_score'] = y_score
        scores['N_1'] = N_1
        scores['N_2'] = N_2
        output = os.path.join(args.svm, 'scores.json')
        with open(output, 'wb') as fp:
            json.dump(scores, fp)

    # y_score = np.asarray(y_score)
    # y_test = np.asarray(y_test)

    plot = 'default'
    f, fpr, tpr = plot_ROC_CIc(y_test, y_score, N_1, N_2)

    if plot == 'default':
        plot = ''

    output = os.path.join(args.svm, 'roc' + plot + '.png')
    f.savefig(output)
    print(("ROC saved as {} !").format(output))

    output = os.path.join(args.svm, 'roc' + plot + '.tex')
    tikz_save(output)

    # Save ROC values as JSON
    output = os.path.join(args.svm, 'roc' + plot + '.csv')
    with open(output, 'wb') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(['FPR', 'TPR'])
        for i in range(0, len(fpr)):
            data = [str(fpr[i]), str(tpr[i])]
            writer.writerow(data)

    print(("ROC saved as {} !").format(output))


def main():
parser = argparse.ArgumentParser(description='Radiomics results')
parser.add_argument('-svm', '--svm', metavar='svm',
                    nargs='+', dest='svm', type=str, required=True,
                    help='SVM file (HDF)')
parser.add_argument('-pinfo', '--pinfo', metavar='pinfo',
                    nargs='+', dest='pinfo', type=str, required=True,
                    help='Patient Info File (txt)')
args = parser.parse_args()

if type(args.svm) is list:
    args.svm = ''.join(args.svm)

if type(args.pinfo) is list:
    args.pinfo = ''.join(args.pinfo)



prediction = pd.read_hdf(args.svm)
n_class = 1
label_type = prediction.keys()[0] #NOTE: Assume we want to have the first key
N_1 = len(prediction[label_type].Y_train[0])
N_2 = len(prediction[label_type].Y_test[0])
_, predictions = plot_multi_SVM(prediction, args.pinfo, label_type, show_plots=False,
                                key=None, n_classifiers=[n_class], alpha=0.95)

y_score = predictions[str(n_class)]['y_score']
y_test = predictions[str(n_class)]['y_test']
plot = 'default'
print(len(y_test), len(y_score))
    # y_score = np.asarray(y_score)
    # y_test = np.asarray(y_test)

    plot = 'default'
    f, fpr, tpr = plot_ROC_CIc(y_tests, y_scores, N_1, N_2)

    if plot == 'default':
        plot = '_' + str(n_class)

    output = os.path.join(args.svm, 'roc' + plot + '.png')
    f.savefig(output)
    print(("ROC saved as {} !").format(output))

    output = os.path.join(args.svm, 'roc' + plot + '.tex')
    tikz_save(output)

    # Save ROC values as JSON
    output = os.path.join(args.svm, 'roc' + plot + '.csv')
    with open(output, 'wb') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(['FPR', 'TPR'])
        for i in range(0, len(fpr)):
            data = [str(fpr[i]), str(tpr[i])]
            writer.writerow(data)

    print(("ROC saved as {} !").format(output))


if __name__ == '__main__':
    main()
