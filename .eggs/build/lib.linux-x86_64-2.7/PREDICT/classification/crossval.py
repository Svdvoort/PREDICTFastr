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
import PREDICT.classification.parameter_optimization as po

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import random


def crossval(config, label_data, image_features,
             classifier, param_grid, use_fastr=False):
    """
    Constructs multiple individual SVMs based on the label settings

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
        image_features (numpy array): The values for the different features


    Returns:
        SVM_data (pandas dataframe)
    """

    patient_IDs = label_data['patient_IDs']
    label_value = label_data['mutation_label']
    label_name = label_data['mutation_name']

    logfilename = os.path.join(os.getcwd(), 'SVM.log')
    print logfilename
    logging.basicConfig(filename=logfilename, level=logging.DEBUG)
    N_iterations = config['CrossValidation']['N_iterations']
    test_size = config['CrossValidation']['test_size']

    svm_labelss = dict()

    print('features')
    logging.debug('Starting SVM')
    print(image_features.shape)
    for i_class, i_name in zip(label_value, label_name):
        i_class = i_class.ravel()

        save_data = list()

        for i in range(0, N_iterations):
            seed = np.random.randint(5000)

            # Split into test and training set, where the percentage of each
            # label is maintained
            X_train, X_test, Y_train, Y_test,\
                patient_ID_train, patient_ID_test\
                = train_test_split(image_features, i_class, patient_IDs,
                                   test_size=test_size, random_state=seed,
                                   stratify=i_class)
            # Scale the features
            scaler = StandardScaler().fit(X_train)
            X_train = scaler.transform(X_train)
            X_test = scaler.transform(X_test)

            # Find best hyperparameters and construct svm
            config['HyperOptimization']['use_fastr'] = use_fastr
            svm = po.random_search_parameters(X_train, Y_train,
                                              **config['HyperOptimization'])

            temp_save_data = (svm, X_train, X_test, Y_train, Y_test,
                              patient_ID_train, patient_ID_test, seed, scaler)

            save_data.append(temp_save_data)

        [svms, X_train_set, X_test_set, Y_train_set, Y_test_set,
         patient_ID_train_set, patient_ID_test_set, seed_set, scalers] =\
            zip(*save_data)

        panda_labels = ['svms', 'X_train', 'X_test', 'Y_train', 'Y_test',
                        'config', 'patient_ID_train', 'patient_ID_test',
                        'random_seed', 'scaler']

        panda_data_temp =\
            pd.Series([svms, X_train_set, X_test_set, Y_train_set,
                       Y_test_set, config, patient_ID_train_set,
                       patient_ID_test_set, seed_set, scalers],
                      index=panda_labels,
                      name='Constructed crossvalidation')

        i_name = ''.join(i_name)
        svm_labelss[i_name] = panda_data_temp

    panda_data = pd.DataFrame(svm_labelss)

    return panda_data
