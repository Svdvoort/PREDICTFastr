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
import fastr


def crossvalfastr(config, label_data, image_features,
                  classifier, param_grid, pathname):
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
        tempdir = fastr.config.mounts['tmp']

        if not os.path.exists(os.path.join(tempdir, pathname)):
            os.makedirs(os.path.join(tempdir, pathname))

        source_files = dict()
        sink_files = list()
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

            # Save to PD series
            source_labels = ['X_train', 'X_test', 'Y_train', 'Y_test',
                             'config', 'patient_ID_train', 'patient_ID_test',
                             'classifier', 'param_grid', 'random_seed',
                             'scaler']

            source_data =\
                pd.Series([X_train, X_test, Y_train,
                           Y_test, config, patient_ID_train,
                           patient_ID_test, classifier, param_grid,
                           seed, scaler],
                          index=source_labels,
                          name='Source Data')
            save_label = ("Source Data Iteration {}").format(i)
            name = ("SD_it_in_{}.hdf5").format(i)
            save_name = os.path.join(tempdir, pathname, name)
            source_data.to_hdf(save_name, save_label)

            output = ("SD_it_out_{}.hdf5").format(i)
            output = os.path.join(tempdir, pathname, output)

            source_files[str(i)] = ('vfs://tmp/{}/{}').format(pathname, name)
            sink_files.append(output)

        # Submit jobs to FASTR
        network = fastr.Network('SVM')
        source_data = network.create_source('HDF5', id_='data')
        sink_svm = network.create_sink('HDF5', id_='SVM')
        svm = network.create_node('OptimizeSVM', memory='8G', id_='SVM_Optimize')

        svm.inputs['data'] = source_data.output
        sink_svm.input = svm.outputs['svm']

        source_data = {'data': source_files}
        sink_data = {'SVM': ("vfs://tmp/{}/SD_it_out_{{sample_id}}.hdf5").format(pathname)}

        network.draw_network(network.id, draw_dimension=True)
        network.execute(source_data, sink_data, tmpdir=os.path.join(tempdir, pathname))

        # Read in the output data once finished
        save_data = list()
        for output in sink_files:
            data = pd.read_hdf(output)

            temp_save_data = (data['svm'], data['X_train'], data['X_test'],
                              data['Y_train'], data['Y_test'],
                              data['patient_ID_train'], data['patient_ID_test'],
                              data['random_seed'], data['scaler'])

            save_data.append(temp_save_data)

        # Convert output to pandas dataframe
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


def crossval(config, label_data, image_features,
             classifier, param_grid, pathname):
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
        tempdir = fastr.config.mounts['tmp']

        if not os.path.exists(os.path.join(tempdir, pathname)):
            os.makedirs(os.path.join(tempdir, pathname))

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
