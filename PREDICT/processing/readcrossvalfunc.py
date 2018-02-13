import glob
import pandas as pd
import os
from PREDICT.trainclassifier import readdata
import json
import PREDICT.IOparser.config_io_classifier as config_io
# from collections import Counter
# import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
import scipy.stats as st
from PREDICT.plotting.compute_CI import compute_confidence as CI
from PREDICT.genetics.genetic_processing import findmutationdata
import natsort
import sklearn
from sklearn.metrics import r2_score, mean_squared_error
import sys
import lifelines as ll
from scipy.stats import pearsonr, spearmanr
from PREDICT.processing.ICC import ICC
import csv
import collections

from PREDICT.processing.fitandscore import fit_and_score
from sklearn.base import clone


def readcrossval(feat_m1, config, sinkfolder, patientinfo, outputfolder,
                 feat_m2=None, feat_m3=None, alpha=0.95, label_type=None,
                 survival=False, n_classifiers=[1, 5, 10]):
    # n_classifiers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 20 ,25, 30, 40 , 50]
    n_classifiers = [1]
    config = config_io.load_config(config)
    sinks = glob.glob(sinkfolder + 'RS*.hdf5')

    # Sort sinks based on creation date
    sinktimes = [os.path.getmtime(f) for f in sinks]
    sinks = [s for _, s in sorted(zip(sinktimes, sinks))]

    if label_type is None:
        label_type = config['Genetics']['mutation_type']

    if survival:
        # Also extract time to event and if event occurs from mutation data
        labels = [label_type, ['E'], ['T']]
    else:
        labels = [[label_type]]

    if feat_m1:
        label_data, _ =\
            readdata(feat_m1, feat_m2, feat_m3, patientinfo,
                     labels)
    else:
        # No feature files found
        label_data, _ = findmutationdata(patientinfo, labels)

    for n_class in n_classifiers:
        output_json = os.path.join(outputfolder, ('performance_{}.json').format(str(n_class)))

        sensitivity = list()
        specificity = list()
        precision = list()
        accuracy = list()
        auc = list()
        # auc_train = list()
        f1_score_list = list()

        patient_classification_list = dict()

        patient_IDs = label_data['patient_IDs']
        mutation_label = label_data['mutation_label']

        trained_classifiers = list()

        y_score = list()
        y_test = list()
        pid_test = list()
        y_predict = list()

        # For SVR
        r2score = list()
        MSE = list()
        coefICC = list()
        PearsonC = list()
        PearsonP = list()
        SpearmanC = list()
        SpearmanP = list()

        if survival:
            cindex = list()
            coxp = list()
            coxcoef = list()

        patient_MSE = dict()

        csvfile = os.path.join(outputfolder, 'scores.csv')
        towrite = list()

        empty_scores = {k: '' for k in natsort.natsorted(patient_IDs)}
        empty_scores = collections.OrderedDict(sorted(empty_scores.items()))
        towrite.append(["Patient"] + empty_scores.keys())
        params = dict()
        for num, s in enumerate(sinks):
            scores = empty_scores.copy()
            print("Processing {} / {}.").format(str(num + 1), str(len(sinks)))
            with open(s, 'r') as fp:
                sr = pd.read_hdf(fp)
            sr = sr['Constructed crossvalidation']
            t = sr.trained_classifier
            trained_classifiers.append(sr.trained_classifier)

            # Extract test info
            test_patient_IDs = sr.patient_ID_test
            X_test = sr.X_test
            Y_test = sr.Y_test

            # Extract sample size
            N_1 = float(len(sr.patient_ID_train))
            N_2 = float(len(sr.patient_ID_test))

            test_indices = list()
            for i_ID in test_patient_IDs:
                test_indices.append(np.where(patient_IDs == i_ID)[0][0])

                if i_ID not in patient_classification_list:
                    patient_classification_list[i_ID] = dict()
                    patient_classification_list[i_ID]['N_test'] = 0
                    patient_classification_list[i_ID]['N_correct'] = 0
                    patient_classification_list[i_ID]['N_wrong'] = 0

                patient_classification_list[i_ID]['N_test'] += 1

            # y_truth = [mutation_label[0][k] for k in test_indices]
            # FIXME: order can be switched, need to find a smart fix
            # 1 for normal, 0 for KM
            y_truth = [mutation_label[0][k][0] for k in test_indices]

            # Predict using the top N classifiers
            results = t.cv_results_['rank_test_score']
            indices = range(0, len(results))
            sortedindices = [x for _, x in sorted(zip(results, indices))]
            sortedindices = sortedindices[0:n_class]
            y_prediction = np.zeros([n_class, len(y_truth)])
            y_score = np.zeros([n_class, len(y_truth)])

            # Get some base objects required
            feature_labels = pd.read_hdf(feat_m1[0]).feature_labels
            base_estimator = t.estimator
            X_train = [(x, feature_labels) for x in sr.X_train]
            y_train = sr.Y_train
            y_train_prediction = np.zeros([n_class, len(y_train)])
            scorer = t.scorer_
            train = np.asarray(range(0, len(y_train)))
            test = train
            del sr # Save some memory
            # cv_iter = list(t.cv.iter(X_train, y_train))

            # NOTE: need to build this in the SearchCVFastr Object
            for i, index in enumerate(sortedindices):
                print("Processing number {} of {} classifiers.").format(str(i + 1), str(n_class))
                X_testtemp = X_test[:]

                # Get the parameters from the index
                parameters_est = t.cv_results_['params'][index]
                parameters_all = t.cv_results_['params_all'][index]

                # NOTE: kernel parameter can be unicode
                kernel = str(parameters_est[u'kernel'])
                del parameters_est[u'kernel']
                del parameters_all[u'kernel']
                parameters_est['kernel'] = kernel
                parameters_all['kernel'] = kernel

                # Refit a classifier using the settings given
                print("Refitting classifier with best settings.")
                best_estimator = clone(base_estimator).set_params(**parameters_est)

                ret, GroupSel, VarSel, SelectModel, feature_labels[0], scaler =\
                    fit_and_score(best_estimator, X_train, y_train, scorer,
                                  train, test, True, parameters_all,
                                  t.fit_params,
                                  t.return_train_score,
                                  True, True, True,
                                  t.error_score)

                X = [x[0] for x in X_train]
                if GroupSel is not None:
                    X = GroupSel.transform(X)
                    X_testtemp = GroupSel.transform(X_testtemp)

                if SelectModel is not None:
                    X = SelectModel.transform(X)
                    X_testtemp = SelectModel.transform(X_testtemp)

                if VarSel is not None:
                    X = VarSel.transform(X)
                    X_testtemp = VarSel.transform(X_testtemp)

                if scaler is not None:
                    X = scaler.transform(X)
                    X_testtemp = scaler.transform(X_testtemp)

                if y_train is not None:
                    best_estimator.fit(X, y_train, **t.fit_params)
                else:
                    best_estimator.fit(X, **t.fit_params)

                # Predict the posterios using the fitted classifier for the training set
                print("Evaluating performance on training set.")
                if hasattr(best_estimator, 'predict_proba'):
                    probabilities = best_estimator.predict_proba(X)
                    y_train_prediction[i, :] = probabilities[:, 1]
                else:
                    # Regression has no probabilities
                    probabilities = best_estimator.predict(X)
                    y_train_prediction[i, :] = probabilities[:]

                # Predict the posterios using the fitted classifier for the test set
                print("Evaluating performance on test set.")
                if hasattr(best_estimator, 'predict_proba'):
                    probabilities = best_estimator.predict_proba(X_testtemp)
                    y_prediction[i, :] = probabilities[:, 1]
                else:
                    # Regression has no probabilities
                    probabilities = best_estimator.predict(X_testtemp)
                    y_prediction[i, :] = probabilities[:]

                if type(t.estimator) == sklearn.svm.classes.SVC:
                    y_score[i, :] = best_estimator.decision_function(X_testtemp)
                else:
                    y_score[i, :] = best_estimator.decision_function(X_testtemp)[:, 0]

                # Add number parameter settings
                for k in parameters_all.keys():
                    if k not in params.keys():
                        params[k] = list()
                    params[k].append(parameters_all[k])

                # Save some memory
                del best_estimator, X, X_testtemp, ret, GroupSel, VarSel, SelectModel, scaler, parameters_est, parameters_all, probabilities

            # Take mean over posteriors of top n
            y_train_prediction_m = np.mean(y_train_prediction, axis=0)
            y_prediction_m = np.mean(y_prediction, axis=0)

            # NOTE: Not sure if this is best way to compute AUC
            y_score = y_prediction_m

            if type(t.estimator) == sklearn.svm.classes.SVC:
                # Look for optimal F1 performance on training set
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
                best_thresh = 0.5
                y_prediction = np.zeros(y_prediction_m.shape)
                for ip, y in enumerate(y_prediction_m):
                    if y > best_thresh:
                        y_prediction[ip] = 1
                    else:
                        y_prediction[ip] = 0

                # y_prediction = t.predict(X_temp)

                y_prediction = [min(max(y, 0), 1) for y in y_prediction]
            else:
                y_prediction = y_prediction_m
                y_prediction = [min(max(y, 0), 1) for y in y_prediction]

            print "Truth: ", y_truth
            print "Prediction: ", y_prediction

            for k, v in zip(test_patient_IDs, y_prediction):
                scores[k] = v

            # for k, v in scores.iteritems():
            #     print k, v
            #
            # raise IOError
            towrite.append(["Iteration " + str()] + scores.values())

            if type(t.estimator) == sklearn.svm.classes.SVC:
                for i_truth, i_predict, i_test_ID in zip(y_truth, y_prediction, test_patient_IDs):
                    if i_truth == i_predict:
                        patient_classification_list[i_test_ID]['N_correct'] += 1
                    else:
                        patient_classification_list[i_test_ID]['N_wrong'] += 1

            if type(t.estimator) == sklearn.svm.classes.SVC:
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
                # y_score = t.decision_function(X_temp)
                auc.append(roc_auc_score(y_truth, y_score))
                f1_score_list.append(f1_score(y_truth, y_prediction, average='weighted'))
            # elif type(t.estimator) == sklearn.svm.classes.SVR:
            else:
                # y_score.extend(svm[k].ix('svms')[0].predict_proba(X_test))
                # y_predict.extend(svm[k].ix('svms')[0].predict(X_test))
                # y_test.extend(Y_test)
                # pid_test.extend(pidt)
                r2score.append(r2_score(y_truth, y_prediction))
                MSE.append(mean_squared_error(y_truth, y_prediction))
                coefICC.append(ICC(np.column_stack((y_prediction, y_truth))))
                C = pearsonr(y_prediction, y_truth)
                PearsonC.append(C[0])
                PearsonP.append(C[1])
                C = spearmanr(y_prediction, y_truth)
                SpearmanC.append(C.correlation)
                SpearmanP.append(C.pvalue)

                if survival:
                    # Extract time to event and event from label data
                    E_truth = np.asarray([mutation_label[1][k][0] for k in test_indices])
                    T_truth = np.asarray([mutation_label[2][k][0] for k in test_indices])

                    # Concordance index
                    cindex.append(1 - ll.utils.concordance_index(T_truth, y_prediction, E_truth))

                    # Fit Cox model using SVR output, time to event and event
                    data = {'predict': y_prediction, 'E': E_truth, 'T': T_truth}
                    data = pd.DataFrame(data=data, index=test_patient_IDs)

                    try:
                        cph = ll.CoxPHFitter()
                        cph.fit(data, duration_col='T', event_col='E')

                        coxcoef.append(cph.summary['coef']['predict'])
                        coxp.append(cph.summary['p']['predict'])
                    except ValueError:
                        # Convergence halted, delta contains nan values?
                        coxcoef.append(1)
                        coxp.append(0)
                    except np.linalg.LinAlgError:
                        #FIXME: Singular matrix
                        coxcoef.append(1)
                        coxp.append(0)

        towrite = zip(*towrite)
        with open(csvfile, 'wb') as csv_file:
            writer = csv.writer(csv_file)
            for w in towrite:
                writer.writerow(w)

        # print(N_1)
        # print(N_2)

        if type(t.estimator) == sklearn.svm.classes.SVC:
            N_iterations = len(sinks)
            accuracy_mean = np.mean(accuracy)
            S_uj = 1.0 / max((N_iterations - 1), 1) * np.sum((accuracy_mean - accuracy)**2.0)

            # print Y_test

            accuracy_var = np.sqrt((1.0/N_iterations + N_2/N_1)*S_uj)
            # print(accuracy_var)
            # print(np.sqrt(1/N_iterations*S_uj))
            # print(st.sem(accuracy))

            stats = dict()
            stats["Accuracy 95%:"] = str(CI(accuracy, N_1, N_2, alpha))

            stats["AUC 95%:"] = str(CI(auc, N_1, N_2, alpha))

            stats["F1-score 95%:"] = str(CI(f1_score_list, N_1, N_2, alpha))

            stats["Precision 95%:"] = str(CI(precision, N_1, N_2, alpha))

            stats["Sensitivity 95%: "] = str(CI(sensitivity, N_1, N_2, alpha))

            stats["Specificity 95%:"] = str(CI(specificity, N_1, N_2, alpha))

            print("Accuracy 95%:" + str(CI(accuracy, N_1, N_2, alpha)))

            print("AUC 95%:" + str(CI(auc, N_1, N_2, alpha)))

            print("F1-score 95%:" + str(CI(f1_score_list, N_1, N_2, alpha)))

            print("Precision 95%:" + str(CI(precision, N_1, N_2, alpha)))

            print("Sensitivity 95%: " + str(CI(sensitivity, N_1, N_2, alpha)))

            print("Specificity 95%:" + str(CI(specificity, N_1, N_2, alpha)))

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
        # elif type(t.estimator) == sklearn.svm.classes.SVR:
        else:
            # Compute confidence intervals from cross validations
            stats = dict()
            stats["r2_score 95%:"] = str(CI(r2score, N_1, N_2, alpha))
            stats["MSE 95%:"] = str(CI(MSE, N_1, N_2, alpha))
            stats["ICC 95%:"] = str(CI(coefICC, N_1, N_2, alpha))
            stats["PearsonC 95%:"] = str(CI(PearsonC, N_1, N_2, alpha))
            stats["SpearmanC 95%: "] = str(CI(SpearmanC, N_1, N_2, alpha))
            stats["PearsonP 95%:"] = str(CI(PearsonP, N_1, N_2, alpha))
            stats["SpearmanP 95%: "] = str(CI(SpearmanP, N_1, N_2, alpha))

            if survival:
                stats["Concordance 95%:"] = str(CI(cindex, N_1, N_2, alpha))
                stats["Cox coef. 95%:"] = str(CI(coxcoef, N_1, N_2, alpha))
                stats["Cox p 95%:"] = str(CI(coxp, N_1, N_2, alpha))

            # Calculate and sort individual patient MSE
            patient_MSE = {k: np.mean(v) for k, v in patient_MSE.iteritems()}
            order = np.argsort(patient_MSE.values())
            sortedkeys = np.asarray(patient_MSE.keys())[order].tolist()
            sortedvalues = np.asarray(patient_MSE.values())[order].tolist()
            patient_MSE = [(k, v) for k, v in zip(sortedkeys, sortedvalues)]

            for p in patient_MSE:
                print p[0], p[1]

            stats["Patient_MSE"] = patient_MSE

            for k, v in stats.iteritems():
                print k, v

        # Check which parameters were most often used
        params = paracheck(params)
        # params = dict()
        # for num, classf in enumerate(trained_classifiers):
        #     params_temp = classf.best_params_
        #     if num == 0:
        #         for k in params_temp.keys():
        #             params[k] = list()
        #             params[k].append(params_temp[k])
        #     else:
        #         for k in params_temp.keys():
        #             params[k].append(params_temp[k])
        #
        # print params

        # # Make histograms or box plots of params
        # for k in params.keys():
        #     para = params[k]
        #     print k
        #     if type(para[0]) is unicode:
        #         letter_counts = Counter(para)
        #         values = letter_counts.values()
        #         keys = letter_counts.keys()
        #         print keys, values
        #         plt.bar(range(len(values)), values, align='center')
        #         plt.xticks(range(len(keys)), keys)
        #         plt.show()
        #     else:
        #         # Make a standard boxplot
        #         plt.figure()
        #         plt.boxplot(para, 0, 'gD')
        #         plt.show()

        # Save output
        savedict = dict()
        savedict["Statistics"] = stats
        savedict['Parameters'] = params

        if type(output_json) is list:
            output_json = ''.join(output_json)

        if not os.path.exists(os.path.dirname(output_json)):
            os.makedirs(os.path.dirname(output_json))

        with open(output_json, 'w') as fp:
            json.dump(savedict, fp, indent=4)

        print("Saved data!")


def paracheck(parameters):
    output = dict()
    # print parameters

    f = parameters['semantic_features']
    total = float(len(f))
    count_semantic = sum([i == 'True' for i in f])
    ratio_semantic = count_semantic/total
    print("Semantic: " + str(ratio_semantic))
    output['semantic_features'] = ratio_semantic

    f = parameters['patient_features']
    count_patient = sum([i == 'True' for i in f])
    ratio_patient = count_patient/total
    print("patient: " + str(ratio_patient))
    output['patient_features'] = ratio_patient

    f = parameters['orientation_features']
    count_orientation = sum([i == 'True' for i in f])
    ratio_orientation = count_orientation/total
    print("orientation: " + str(ratio_orientation))
    output['orientation_features'] = ratio_orientation

    f = parameters['histogram_features']
    count_histogram = sum([i == 'True' for i in f])
    ratio_histogram = count_histogram/total
    print("histogram: " + str(ratio_histogram))
    output['histogram_features'] = ratio_histogram

    f = parameters['shape_features']
    count_shape = sum([i == 'True' for i in f])
    ratio_shape = count_shape/total
    print("shape: " + str(ratio_shape))
    output['shape_features'] = ratio_shape

    if 'coliage_features' in parameters.keys():
        f = parameters['coliage_features']
        count_coliage = sum([i == 'True' for i in f])
        ratio_coliage = count_coliage/total
        print("coliage: " + str(ratio_coliage))
        output['coliage_features'] = ratio_coliage

    if 'phase_features' in parameters.keys():
        f = parameters['phase_features']
        count_phase = sum([i == 'True' for i in f])
        ratio_phase = count_phase/total
        print("phase: " + str(ratio_phase))
        output['phase_features'] = ratio_phase

    if 'vessel_features' in parameters.keys():
        f = parameters['vessel_features']
        count_vessel = sum([i == 'True' for i in f])
        ratio_vessel = count_vessel/total
        print("vessel: " + str(ratio_vessel))
        output['vessel_features'] = ratio_vessel

    if 'log_features' in parameters.keys():
        f = parameters['log_features']
        count_log = sum([i == 'True' for i in f])
        ratio_log = count_log/total
        print("log: " + str(ratio_log))
        output['log_features'] = ratio_log

    f = parameters['texture_features']
    count_texture_all = sum([i == 'True' for i in f])
    ratio_texture_all = count_texture_all/total
    print("texture_all: " + str(ratio_texture_all))
    output['texture_all_features'] = ratio_texture_all

    count_texture_no = sum([i == 'False' for i in f])
    ratio_texture_no = count_texture_no/total
    print("texture_no: " + str(ratio_texture_no))
    output['texture_no_features'] = ratio_texture_no

    count_texture_Gabor = sum([i == 'Gabor' for i in f])
    ratio_texture_Gabor = count_texture_Gabor/total
    print("texture_Gabor: " + str(ratio_texture_Gabor))
    output['texture_Gabor_features'] = ratio_texture_Gabor

    count_texture_LBP = sum([i == 'LBP' for i in f])
    ratio_texture_LBP = count_texture_LBP/total
    print("texture_LBP: " + str(ratio_texture_LBP))
    output['texture_LBP_features'] = ratio_texture_LBP

    count_texture_GLCM = sum([i == 'GLCM' for i in f])
    ratio_texture_GLCM = count_texture_GLCM/total
    print("texture_GLCM: " + str(ratio_texture_GLCM))
    output['texture_GLCM_features'] = ratio_texture_GLCM

    count_texture_GLRLM = sum([i == 'GLRLM' for i in f])
    ratio_texture_GLRLM = count_texture_GLRLM/total
    print("texture_GLRLM: " + str(ratio_texture_GLRLM))
    output['texture_GLRLM_features'] = ratio_texture_GLRLM

    count_texture_GLSZM = sum([i == 'GLSZM' for i in f])
    ratio_texture_GLSZM = count_texture_GLSZM/total
    print("texture_GLSZM: " + str(ratio_texture_GLSZM))
    output['texture_GLSZM_features'] = ratio_texture_GLSZM

    f = parameters['degree']
    print("Polynomial Degree: " + str(np.mean(f)))
    output['polynomial_degree'] = np.mean(f)

    return output

if __name__ == '__main__':
    # config = '/scratch/mstarmans/tmp/BLT_Razvan_1115/config_classification/id_0/result/config_BLT_Razvan_1115_0.ini'
    # sinkfolder = '/scratch/mstarmans/tmp/GSoutBLT/*'
    # outputfolder = '/scratch/mstarmans/tmp/BLT_Razvan_1115/classify/all'
    # patientinfo = '/scratch/mstarmans/tmp/BLT_Razvan_1115/patientclass/id_0/result/pinfo_BLT.txt'
    # feat_m1 = glob.glob('/archive/mstarmans/Output/BLT_Razvan_1115/features*.hdf5')
    # label_type = [['GP']]

    # config = '/scratch/mstarmans/tmp/CLM_M1_SVR_0811/config_classification/id_0/result/config_CLM_M1_SVR_0811_0.ini'
    # sinkfolder = '/scratch/mstarmans/tmp/GSout/*'
    # outputfolder = '/scratch/mstarmans/tmp/CLM_M1_SVR_0811/classify/all'
    # patientinfo = '/scratch/mstarmans/tmp/CLM_M1_SVR_0811/patientclass/id_0/result/pinfo_CLM_KM.txt'
    # feat_m1 = glob.glob('/archive/mstarmans/Output/CLM_M1_SVR_0811/features*.hdf5')

    # config = '/scratch/mstarmans/tmp/DM/config_DM_T41A.ini'
    # sinkfolder = '/scratch/mstarmans/tmp/GSout_T41A22/*'
    # outputfolder = '/scratch/mstarmans/tmp/DM/classify/22'
    # patientinfo = '/scratch/mstarmans/tmp/DM/patientclass/id_0/result/pinfo_DM.txt'
    # feat_m1 = glob.glob('/archive/mstarmans/Output/DM/features*.hdf5')
    # label_type = [['T41A']]

    # config = '/scratch/mstarmans/tmp/CLM_MICCAI_trainall/config_CLM_MICCAI_trainall_0.ini'
    # sinkfolder = '/scratch/mstarmans/tmp/GSout_CPM_SVRall/*'
    # outputfolder = '/scratch/mstarmans/tmp/CLM_MICCAI_trainall/classify/all'
    # patientinfo = '/scratch/mstarmans/tmp/CLM_MICCAI_trainall/patientclass/id_0/result/pinfo_CLM_MICCAI_traintest_months.txt'
    # feat_m1 = glob.glob('/archive/mstarmans/Output/CLM_MICCAI/features*.hdf5')
    # label_type = [['KM']]

    # CPM: Test vs train
    # config = '/scratch/mstarmans/tmp/CLM_MICCAI_testvstrain/config_CLM_MICCAI_testvstrain_0.ini'
    # sinkfolder = '/scratch/mstarmans/tmp/GSout/*'
    # outputfolder = '/scratch/mstarmans/tmp/CLM_MICCAI_testvstrain/classify/all'
    # patientinfo = '/scratch/mstarmans/tmp/CLM_MICCAI_testvstrain/patientclass/id_0/result/pinfo_CLM_MICCAI_traintest_months.txt'
    # feat_m1 = glob.glob('/archive/mstarmans/Output/CLM_MICCAI/features*.hdf5')
    # label_type = [['test']]

    # # CPM all
    config = '/scratch/mstarmans/tmp/CLM_MICCAI_trainall/config_CLM_MICCAI_trainall_0.ini'
    sinkfolder = '/archive/mstarmans/Output/CLM_MICCAI/FixedSplits/GSoutNoSem/*'
    # sinkfolder = '/archive/mstarmans/Output/CLM_MICCAI/ElasticNet_part/'
    outputfolder = '/archive/mstarmans/Output/CLM_MICCAI/FixedSplits/GSoutNoSem'
    # outputfolder = sinkfolder
    patientinfo = '/scratch/mstarmans/tmp/CLM_MICCAI_trainall/patientclass/id_0/result/MICCAI_CPM_Liver_Mets_All_Ground_truth.txt'
    feat_m1 = glob.glob('/archive/mstarmans/Output/CLM_MICCAI/features*.hdf5')
    label_type = [['KM']]
    survival = True

    # # CPM chemo vs nonchemo
    # config = '/scratch/mstarmans/tmp/CLM_MICCAI_nonewfeat_0207_chemo/config_CLM_MICCAI_nonewfeat_0207_chemo_0.ini'
    # sinkfolder = '/archive/mstarmans/Output/CLM_MICCAI/FixedSplits/GSOutNoNewChemo/*'
    # # sinkfolder = '/archive/mstarmans/Output/CLM_MICCAI/ElasticNet_part/'
    # outputfolder = '/archive/mstarmans/Output/CLM_MICCAI/FixedSplits/GSOutNoNewChemo'
    # # outputfolder = sinkfolder
    # patientinfo = '/scratch/mstarmans/tmp/CLM_MICCAI_nonewfeat_0207_chemo/patientclass/id_0/result/MICCAI_CPM_Liver_Mets_All_Ground_truth.txt'
    # feat_m1 = glob.glob('/archive/mstarmans/Data/CLM_MICCAI_Both/features*.hdf5')
    # label_type = [['KM']]
    # survival = True

    # # IC
    # config = '/scratch/mstarmans/tmp/IC_1201/config_IC_1201_0.ini'
    # sinkfolder = '/scratch/mstarmans/tmp/GSoutIC/*'
    # outputfolder = '/scratch/mstarmans/tmp/IC_1201/classify/all'
    # patientinfo = '/scratch/mstarmans/tmp/IC_1201/patientclass/id_0/result/pinfo_IC.txt'
    # feat_m1 = glob.glob('/archive/mstarmans/Output/IC_1201/features*.hdf5')
    # label_type = [['prev_stroke_2016']]
    # survival = False

    # # BLT
    # config = '/scratch/mstarmans/tmp/BLT_Razvan_1115/config_BLT_Razvan_1115_0.ini'
    # sinkfolder = '/archive/mstarmans/Output/BLT_Razvan_CV_0728/GSoutBLT/*'
    # outputfolder = '/archive/mstarmans/Output/BLT_Razvan_CV_0728/GSoutBLT'
    # patientinfo = '/scratch/mstarmans/tmp/BLT_Razvan_1115/patientclass/id_0/result/pinfo_BLT.txt'
    # feat_m1 = glob.glob('/archive/mstarmans/Output/BLT_Razvan_CV_0728/features*.hdf5')
    # label_type = [['GP']]

    # # CLM m1
    # config = '/scratch/mstarmans/tmp/CLM_m1_1221/config_CLM_m1_1221_0.ini'
    # sinkfolder = '/scratch/mstarmans/tmp/GSoutm1/*'
    # outputfolder = '/scratch/mstarmans/tmp/GSoutm1'
    # patientinfo = '/scratch/mstarmans/tmp/CLM_m1_1221/patientclass/id_0/result/pinfo_CLM_KM.txt'
    # feat_m1 = glob.glob('/archive/mstarmans/Output/CLM_m1_1221/features*.hdf5')
    # label_type = [['GP']]

    # # CLM m2
    # config = '/scratch/mstarmans/tmp/CLM_m1_1221/config_CLM_m1_1221_0.ini'
    # sinkfolder = '/scratch/mstarmans/tmp/GSoutm1/*'
    # outputfolder = '/scratch/mstarmans/tmp/GSoutm1'
    # patientinfo = '/scratch/mstarmans/tmp/CLM_m1_1221/patientclass/id_0/result/pinfo_CLM_KM.txt'
    # feat_m1 = glob.glob('/archive/mstarmans/Output/CLM_m1_1221/features*.hdf5')
    # label_type = [['GP']]
    # survival = False
    readcrossval(config=config, sinkfolder=sinkfolder,
                 outputfolder=outputfolder, patientinfo=patientinfo,
                 feat_m1=feat_m1, label_type=label_type,
                 survival=survival)
