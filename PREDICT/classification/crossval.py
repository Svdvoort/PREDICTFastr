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
import pandas as pd
import logging
import os
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.utils import check_random_state
import sklearn
import xlrd
import natsort
import PREDICT.classification.parameter_optimization as po
import PREDICT.addexceptions as ae
from imblearn.over_sampling import RandomOverSampler
import random


def crossval(config, label_data, image_features,
             classifier, param_grid={}, use_fastr=False,
             fastr_plugin=None, tempsave=False,
             fixedsplits=None, ensemble={'Use': False}, outputfolder=None,
             modus='singlelabel'):
    """
    Constructs multiple individual classifiers based on the label settings

    Parameters
    ----------
    config: dict, mandatory
            Dictionary with config settings. See the Github Wiki for the
            available fields and formatting.

    label_data: dict, mandatory
            Should contain the following:
            patient_IDs (list): IDs of the patients, used to keep track of test and
                     training sets, and genetic data
            mutation_label (list): List of lists, where each list contains the
                                   mutations status for that patient for each
                                   mutations
            mutation_name (list): Contains the different mutations that are stored
                                  in the mutation_label

    image_features: numpy array, mandatory
            Consists of a tuple of two lists for each patient:
            (feature_values, feature_labels)

    classifier: sklearn classifier
            The untrained classifier used for training.

    param_grid: dictionary, optional
            Contains the parameters and their values wich are used in the
            grid or randomized search hyperparamater optimization. See the
            construct_classifier function for some examples.

    use_fastr: boolean, default False
            If False, parallel execution through Joblib is used for fast
            execution of the hyperparameter optimization. Especially suited
            for execution on mutlicore (H)PC's. The settings used are
            specified in the config.ini file in the IOparser folder, which you
            can adjust to your system.

            If True, fastr is used to split the hyperparameter optimization in
            separate jobs. Parameters for the splitting can be specified in the
            config file. Especially suited for clusters.

    fastr_plugin: string, default None
            Determines which plugin is used for fastr executions.
            When None, uses the default plugin from the fastr config.

    tempsave: boolean, default False
            If True, create a .hdf5 file after each cross validation containing
            the classifier and results from that that split. This is written to
            the GSOut folder in your fastr output mount. If False, only
            the result of all combined cross validations will be saved to a .hdf5
            file. This will also be done if set to True.

    fixedsplits: string, optional
            By default, random split cross validation is used to train and
            evaluate the machine learning methods. Optionally, you can provide
            a .xlsx file containing fixed splits to be used. See the Github Wiki
            for the format.

    ensemble: dictionary, optional
            Contains the configuration for constructing an ensemble.

    modus: string, default 'singlelabel'
            Determine whether one-vs-all classification (or regression) for
            each single label is used ('singlelabel') or if multilabel
            classification is performed ('multilabel').

    Returns
    ----------
    panda_data: pandas dataframe
            Contains all information on the trained classifier.

    """
    if tempsave:
        import fastr

    patient_IDs = label_data['patient_IDs']
    label_value = label_data['mutation_label']
    label_name = label_data['mutation_name']

    if outputfolder is None:
        logfilename = os.path.join(os.getcwd(), 'classifier.log')
    else:
        logfilename = os.path.join(outputfolder, 'classifier.log')
    print("Logging to file " + str(logfilename))

    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    logging.basicConfig(filename=logfilename, level=logging.DEBUG)
    N_iterations = config['CrossValidation']['N_iterations']
    test_size = config['CrossValidation']['test_size']

    classifier_labelss = dict()

    print('features')
    logging.debug('Starting classifier')
    print(len(image_features))

    # We only need one label instance, assuming they are all the sample
    feature_labels = image_features[0][1]

    # Check if we need to use fixedsplits:
    if fixedsplits is not None and '.xlsx' in fixedsplits:
        # fixedsplits = '/home/mstarmans/Settings/RandomSufflingOfData.xlsx'
        wb = xlrd.open_workbook(fixedsplits)
        wb = wb.sheet_by_index(1)

    if modus == 'singlelabel':
        print('Performing Single class classification.')
        logging.debug('Performing Single class classification.')
    elif modus == 'multilabel':
        print('Performing Multi label classification.')
        logging.debug('Performing Multi class classification.')
        label_value = [label_value]
        label_name = [label_name]
    else:
        m = ('{} is not a valid modus!').format(modus)
        logging.debug(m)
        raise ae.PREDICTKeyError(m)

    for i_class, i_name in zip(label_value, label_name):
        if modus == 'singlelabel':
            i_class_temp = i_class.ravel()

        save_data = list()

        for i in range(0, N_iterations):
            print(('Cross validation iteration {} / {} .').format(str(i + 1), str(N_iterations)))
            logging.debug(('Cross validation iteration {} / {} .').format(str(i + 1), str(N_iterations)))
            random_seed = np.random.randint(5000)
            random_state = check_random_state(random_seed)

            # Split into test and training set, where the percentage of each
            # label is maintained
            if type(classifier) == sklearn.svm.classes.SVR:
                # We cannot do a stratified shuffle split with regression
                stratify = None
            else:
                if modus == 'singlelabel':
                    stratify = i_class_temp
                elif modus == 'multilabel':
                    # Create a stratification object from the labels
                    # Label = 0 means no label equals one
                    # Other label numbers refer to the label name that is 1
                    stratify = list()
                    for pnum in range(0, len(i_class[0])):
                        plabel = 0
                        for lnum, slabel in enumerate(i_class):
                            if slabel[pnum] == 1:
                                plabel = lnum + 1
                        stratify.append(plabel)

                    # Sklearn multiclass requires rows to be objects/patients
                    # i_class = i_class.reshape(i_class.shape[1], i_class.shape[0])
                    i_class_temp = np.zeros((i_class.shape[1], i_class.shape[0]))
                    for n_patient in range(0, i_class.shape[1]):
                        for n_label in range(0, i_class.shape[0]):
                            i_class_temp[n_patient, n_label] = i_class[n_label, n_patient]
                    i_class_temp = i_class_temp
                else:
                    raise ae.PREDICTKeyError('{} is not a valid modus!').format(modus)

            if fixedsplits is None:
                # Use Random Split. Split per patient, not per sample
                unique_patient_IDs, unique_indices =\
                    np.unique(np.asarray(patient_IDs), return_index=True)
                if type(classifier) == sklearn.svm.classes.SVR:
                    unique_stratify = None
                else:
                    unique_stratify = [stratify[i] for i in unique_indices]

                try:
                    unique_PID_train, indices_PID_test\
                        = train_test_split(unique_patient_IDs,
                                           test_size=test_size,
                                           random_state=random_seed,
                                           stratify=unique_stratify)
                except ValueError as e:
                    e = str(e) + ' Increase the size of your validation set.'
                    raise ae.PREDICTValueError(e)

                # Check for all IDs if they are in test or training
                indices_train = list()
                indices_test = list()
                patient_ID_train = list()
                patient_ID_test = list()
                for num, pid in enumerate(patient_IDs):
                    if pid in unique_PID_train:
                        indices_train.append(num)

                        # Make sure we get a unique ID
                        if pid in patient_ID_train:
                            n = 1
                            while str(pid + '_' + str(n)) in patient_ID_train:
                                n += 1
                            pid = str(pid + '_' + str(n))
                        patient_ID_train.append(pid)
                    else:
                        indices_test.append(num)

                        # Make sure we get a unique ID
                        if pid in patient_ID_test:
                            n = 1
                            while str(pid + '_' + str(n)) in patient_ID_test:
                                n += 1
                            pid = str(pid + '_' + str(n))
                        patient_ID_test.append(pid)

                # Split features and labels accordingly
                X_train = [image_features[i] for i in indices_train]
                X_test = [image_features[i] for i in indices_test]
                if modus == 'singlelabel':
                    Y_train = i_class_temp[indices_train]
                    Y_test = i_class_temp[indices_test]
                elif modus == 'multilabel':
                    Y_train = i_class_temp[indices_train, :]
                    Y_test = i_class_temp[indices_test, :]
                else:
                    raise ae.PREDICTKeyError('{} is not a valid modus!').format(modus)

            else:
                # Use pre defined splits
                indices = wb.col_values(i)
                indices = [int(j) for j in indices[1:]]  # First element is "Iteration x"
                train = indices[0:121]
                test = indices[121:]

                # Convert the numbers to the correct indices
                ind_train = list()
                for j in train:
                    success = False
                    for num, p in enumerate(patient_IDs):
                        if str(j).zfill(3) == p[0:3]:
                            ind_train.append(num)
                            success = True
                    if not success:
                        print natsort.natsorted(patient_IDs)
                        raise ae.PREDICTIOError("Patient " + str(j).zfill(3) + " is not included!")

                ind_test = list()
                for j in test:
                    success = False
                    for num, p in enumerate(patient_IDs):
                        if str(j).zfill(3) == p[0:3]:
                            ind_test.append(num)
                            success = True
                    if not success:
                        print natsort.natsorted(patient_IDs)
                        raise ae.PREDICTIOError("Patient " + str(j).zfill(3) + " is not included!")

                X_train = np.asarray(image_features)[ind_train].tolist()
                Y_train = np.asarray(i_class_temp)[ind_train].tolist()
                patient_ID_train = patient_IDs[ind_train]
                X_test = np.asarray(image_features)[ind_test].tolist()
                Y_test = np.asarray(i_class_temp)[ind_test].tolist()
                patient_ID_test = patient_IDs[ind_test]

            X_train, Y_train, trained_classifier =\
                singleiteration(X_train=X_train,
                                Y_train=Y_train,
                                PID_train=patient_ID_train,
                                feature_labels=feature_labels,
                                classifier=classifier,
                                param_grid=param_grid,
                                config_hyperopt=config['HyperOptimization'],
                                use_SMOTE=config['SampleProcessing']['SMOTE'],
                                SMOTE_ratio=config['SampleProcessing']['SMOTE_ratio'],
                                SMOTE_neighbors=config['SampleProcessing']['SMOTE_neighbors'],
                                n_cores=config['General']['Joblib_ncores'],
                                N_jobs=config['General']['Joblib_ncores'],
                                random_state=random_state,
                                use_fastr=use_fastr,
                                fastr_plugin=fastr_plugin,
                                use_ensemble=ensemble['Use'],
                                use_oversampling=config['SampleProcessing']['Oversampling'])

            # We only want to save the feature values and one label array
            X_train = [x[0] for x in X_train]
            X_test = [x[0] for x in X_test]

            temp_save_data = (trained_classifier, X_train, X_test, Y_train,
                              Y_test, patient_ID_train, patient_ID_test, random_seed)

            save_data.append(temp_save_data)

            # Create a temporary save
            if tempsave:
                panda_labels = ['trained_classifier', 'X_train', 'X_test', 'Y_train', 'Y_test',
                                'config', 'patient_ID_train', 'patient_ID_test',
                                'random_seed']

                panda_data_temp =\
                    pd.Series([trained_classifier, X_train, X_test, Y_train,
                               Y_test, config, patient_ID_train,
                               patient_ID_test, random_seed],
                              index=panda_labels,
                              name='Constructed crossvalidation')

                panda_data = pd.DataFrame(panda_data_temp)
                n = 0
                filename = os.path.join(fastr.config.mounts['tmp'], 'GSout', 'RS_' + str(i) + '.hdf5')
                while os.path.exists(filename):
                    n += 1
                    filename = os.path.join(fastr.config.mounts['tmp'], 'GSout', 'RS_' + str(i + n) + '.hdf5')

                if not os.path.exists(os.path.dirname(filename)):
                    os.makedirs(os.path.dirname(filename))

                panda_data.to_hdf(filename, 'SVMdata')
                del panda_data, panda_data_temp

        [classifiers, X_train_set, X_test_set, Y_train_set, Y_test_set,
         patient_ID_train_set, patient_ID_test_set, seed_set] =\
            zip(*save_data)

        panda_labels = ['classifiers', 'X_train', 'X_test', 'Y_train', 'Y_test',
                        'config', 'patient_ID_train', 'patient_ID_test',
                        'random_seed', 'feature_labels']

        panda_data_temp =\
            pd.Series([classifiers, X_train_set, X_test_set, Y_train_set,
                       Y_test_set, config, patient_ID_train_set,
                       patient_ID_test_set, seed_set, feature_labels],
                      index=panda_labels,
                      name='Constructed crossvalidation')

        if modus == 'singlelabel':
            i_name = ''.join(i_name)
        elif modus == 'multilabel':
            i_name = ','.join(i_name)

        classifier_labelss[i_name] = panda_data_temp

    panda_data = pd.DataFrame(classifier_labelss)

    return panda_data


def nocrossval(config, label_data_train, label_data_test, image_features_train,
               image_features_test, classifier, param_grid, use_fastr=False,
               fastr_plugin=None, ensemble={'Use': False},
               modus='singlelabel'):
    """
    Constructs multiple individual classifiers based on the label settings

    Arguments:
        config (Dict): Dictionary with config settings
        label_data (Dict): should contain:
        patient_IDs (list): IDs of the patients, used to keep track of test and
                 training sets, and genetic data
        mutation_label (list): List of lists, where each list contains the
                               mutations status for that patient for each
                               mutations
        mutation_name (list): Contains the different mutations that are stored
                              in the mutation_label
        image_features (numpy array): Consists of a tuple of two lists for each patient:
                                    (feature_values, feature_labels)

        ensemble: dictionary, optional
                Contains the configuration for constructing an ensemble.

        modus: string, default 'singlelabel'
                Determine whether one-vs-all classification (or regression) for
                each single label is used ('singlelabel') or if multilabel
                classification is performed ('multilabel').

    Returns:
        classifier_data (pandas dataframe)
    """

    patient_IDs_train = label_data_train['patient_IDs']
    label_value_train = label_data_train['mutation_label']
    label_name_train = label_data_train['mutation_name']

    patient_IDs_test = label_data_test['patient_IDs']
    if 'mutation_label' in label_data_test.keys():
        label_value_test = label_data_test['mutation_label']
    else:
        label_value_test = [None] * len(patient_IDs_test)

    logfilename = os.path.join(os.getcwd(), 'classifier.log')
    logging.basicConfig(filename=logfilename, level=logging.DEBUG)

    classifier_labelss = dict()

    print('features')
    logging.debug('Starting classifier')
    print(len(image_features_train))

    # Determine modus
    if modus == 'singlelabel':
        print('Performing Single class classification.')
        logging.debug('Performing Single class classification.')
    elif modus == 'multilabel':
        print('Performing Multi label classification.')
        logging.debug('Performing Multi class classification.')
        label_name_train = [label_name_train]
    else:
        m = ('{} is not a valid modus!').format(modus)
        logging.debug(m)
        raise ae.PREDICTKeyError(m)

    # We only need one label instance, assuming they are all the sample
    feature_labels = image_features_train[0][1]
    for i_name in label_name_train:

        save_data = list()

        random_seed = np.random.randint(5000)
        random_state = check_random_state(random_seed)

        # Split into test and training set, where the percentage of each
        # label is maintained
        X_train = image_features_train
        X_test = image_features_test
        if modus == 'singlelabel':
            Y_train = label_value_train.ravel()
            Y_test = label_value_test.ravel()
        else:
            # Sklearn multiclass requires rows to be objects/patients
            Y_train = label_value_train
            Y_train_temp = np.zeros((Y_train.shape[1], Y_train.shape[0]))
            for n_patient in range(0, Y_train.shape[1]):
                for n_label in range(0, Y_train.shape[0]):
                    Y_train_temp[n_patient, n_label] = Y_train[n_label, n_patient]
            Y_train = Y_train_temp

            Y_test = label_value_test
            Y_test_temp = np.zeros((Y_test.shape[1], Y_test.shape[0]))
            for n_patient in range(0, Y_test.shape[1]):
                for n_label in range(0, Y_test.shape[0]):
                    Y_test_temp[n_patient, n_label] = Y_test[n_label, n_patient]
            Y_test = Y_test_temp

        X_train, Y_train, trained_classifier =\
            singleiteration(X_train=X_train,
                            Y_train=Y_train,
                            PID_train=patient_IDs_train,
                            feature_labels=feature_labels,
                            classifier=classifier,
                            param_grid=param_grid,
                            config_hyperopt=config['HyperOptimization'],
                            use_SMOTE=config['SampleProcessing']['SMOTE'],
                            SMOTE_ratio=config['SampleProcessing']['SMOTE_ratio'],
                            SMOTE_neighbors=config['SampleProcessing']['SMOTE_neighbors'],
                            n_cores=config['General']['Joblib_ncores'],
                            N_jobs=config['General']['Joblib_ncores'],
                            random_state=random_state,
                            use_fastr=use_fastr,
                            fastr_plugin=fastr_plugin,
                            use_ensemble=ensemble['Use'],
                            use_oversampling=config['SampleProcessing']['Oversampling'])

        # Extract the feature values
        X_train = np.asarray([x[0] for x in X_train])
        X_test = np.asarray([x[0] for x in X_test])

        temp_save_data = (trained_classifier, X_train, X_test, Y_train,
                          Y_test, patient_IDs_train, patient_IDs_test, random_seed)

        save_data.append(temp_save_data)

        [classifiers, X_train_set, X_test_set, Y_train_set, Y_test_set,
         patient_ID_train_set, patient_ID_test_set, seed_set] =\
            zip(*save_data)

        panda_labels = ['classifiers', 'X_train', 'X_test', 'Y_train', 'Y_test',
                        'config', 'patient_ID_train', 'patient_ID_test',
                        'random_seed', 'feature_labels']

        panda_data_temp =\
            pd.Series([classifiers, X_train_set, X_test_set, Y_train_set,
                       Y_test_set, config, patient_ID_train_set,
                       patient_ID_test_set, seed_set, feature_labels],
                      index=panda_labels,
                      name='Constructed crossvalidation')

        i_name = ''.join(i_name)
        classifier_labelss[i_name] = panda_data_temp

    panda_data = pd.DataFrame(classifier_labelss)

    return panda_data


def singleiteration(X_train, Y_train, PID_train, feature_labels,
                    classifier, param_grid, config_hyperopt, use_SMOTE=False,
                    SMOTE_ratio=1, SMOTE_neighbors=10, n_cores=4,
                    N_jobs=4, random_state=None, use_fastr=False,
                    fastr_plugin='LinearExecution',
                    use_ensemble=False, use_oversampling=True):
    '''
    Perform a single iteration of a cross validation.
    '''

    if random_state is None:
        random_seed = np.random.randint(5000)
        random_state = check_random_state(random_seed)

    if use_SMOTE:
        pos_initial = int(np.sum(Y_train))
        neg_initial = int(len(Y_train) - pos_initial)
        len_in = len(Y_train)
        for num, x in enumerate(X_train):
            if num == 0:
                X_train_temp = np.zeros((len(x[0]), 1))
                X_train_temp[:, 0] = np.asarray(x[0])
            else:
                xt = np.zeros((len(x[0]), 1))
                xt[:, 0] = np.asarray(x[0])
                X_train_temp = np.column_stack((X_train_temp, xt))

        X_train_temp = np.transpose(X_train_temp)
        sm = SMOTE(random_state=random_state,
                   ratio=SMOTE_ratio,
                   m_neighbors=SMOTE_neighbors,
                   kind='borderline1',
                   n_jobs=N_jobs)

        # First, replace the NaNs:
        for pnum, (pid, X) in enumerate(zip(PID_train, X_train_temp)):
            for fnum, (f, l) in enumerate(zip(X, feature_labels)):
                if np.isnan(f):
                    print("[PREDICT WARNING] NaN found, patient {}, label {}. Replacing with zero.").format(pid, l)
                    X_train_temp[pnum, fnum] = 0

        X_train, Y_train = sm.fit_sample(X_train_temp, Y_train)
        X_train = [(x.tolist(), feature_labels) for x in X_train]
        pos = int(np.sum(Y_train))
        neg = int(len(Y_train) - pos)
        message = ("Sampling with SMOTE from {} ({} pos, {} neg) to {} ({} pos, {} neg) patients.").format(str(len_in),
                                                                                                           str(pos_initial),
                                                                                                           str(neg_initial),
                                                                                                           str(len(Y_train)),
                                                                                                           str(pos),
                                                                                                           str(neg))
        print(message)

    if use_oversampling:
        print('Oversample underrepresented classes in training.')
        # Oversample underrepresented classes in training
        # We always use a factor 1, e.g. all classes end up with an
        # equal number of samples
        if len(Y_train.shape) == 1:
            # Single Class, use imblearn oversampling

            # Create another random state
            random_seed2 = np.random.randint(5000)
            random_state2 = check_random_state(random_seed2)

            ros = RandomOverSampler(random_state=random_state2)
            X_train, Y_train = ros.fit_sample(X_train, Y_train)

        else:
            # Multi class, use own method as imblearn cannot do this
            sumclass = [np.sum(Y_train[:, i]) for i in range(Y_train.shape[1])]
            maxclass = np.argmax(sumclass)
            for i in range(Y_train.shape[1]):
                if i != maxclass:
                    # Oversample
                    nz = np.nonzero(Y_train[:, i])[0]
                    noversample = sumclass[maxclass] - sumclass[i]
                    while noversample > 0:
                        n_sample = random.randint(0, len(nz) - 1)
                        n_sample = nz[n_sample]
                        i_sample = Y_train[n_sample, :]
                        x_sample = X_train[n_sample]
                        Y_train = np.vstack((Y_train, i_sample))
                        X_train.append(x_sample)
                        noversample -= 1

    # Find best hyperparameters and construct classifier
    config_hyperopt['use_fastr'] = use_fastr
    config_hyperopt['fastr_plugin'] = fastr_plugin
    trained_classifier = po.random_search_parameters(features=X_train,
                                                     labels=Y_train,
                                                     classifier=classifier,
                                                     param_grid=param_grid,
                                                     n_cores=n_cores,
                                                     **config_hyperopt)

    # Create an ensemble if required
    if use_ensemble:
        trained_classifier.create_ensemble(X_train, Y_train)

    return X_train, Y_train, trained_classifier
