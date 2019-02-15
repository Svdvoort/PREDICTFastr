#!/usr/bin/env python

# Copyright 2017-2018 Biomedical Imaging Group Rotterdam, Departments of
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

import xnat
import requests
from requests.packages.urllib3.exceptions import InsecureRequestWarning
import numpy as np
import os
import configparser
import PREDICT.addexceptions as ae


def load_mutation_status(genetic_file, mutation_type):
    """Loads the mutation data from a genetic file

    Args:
        genetic_file (string): The path to the genetic file
        mutation_type (list): List of the genetic mutations to load

    Returns:
        dict: A dict containing 'patient_IDs', 'mutation_label' and
         'mutation_type'
    """
    _, extension = os.path.splitext(genetic_file)
    if extension == '.txt':
        mutation_names, patient_IDs, mutation_status = load_genetic_file(
            genetic_file)
    elif extension == '.ini':
        mutation_names, patient_IDs, mutation_status = load_genetic_XNAT(
            genetic_file)
    else:
        raise ae.PREDICTIOError(extension + ' is not valid genetic file extension.')

    print("Label names to extract: " + str(mutation_type))
    mutation_label = list()
    for i_mutation in mutation_type:
        mutation_index = np.where(mutation_names == i_mutation)[0]
        if mutation_index.size == 0:
            raise ae.PREDICTValueError('Could not find mutation: ' + str(i_mutation))
        else:
            mutation_label.append(mutation_status[:, mutation_index])

    mutation_data = dict()
    mutation_data['patient_IDs'] = patient_IDs
    mutation_data['mutation_label'] = mutation_label
    mutation_data['mutation_name'] = mutation_type

    return mutation_data


def load_genetic_file(input_file):
    """
    Load the patient IDs and genetic data from the genetic file

    Args:
        input_file (string): Path of the genetic file

    Returns:
        mutation_names (numpy array): Names of the different genetic mutations
        patient_ID (numpy array): IDs of patients for which genetic data is
         loaded
        mutation_status (numpy array): The status of the different mutations
         for each patient
    """
    data = np.loadtxt(input_file, np.str)

    # Load and check the header
    header = data[0, :]
    if header[0] != 'Patient':
        raise ae.PREDICTAssertionError('First column should be patient ID!')
    else:
        # cut out the first header, only keep genetic header
        mutation_names = header[1::]

    # Patient IDs are stored in the first column
    patient_ID = data[1:, 0]

    # Mutation status is stored in all remaining columns
    mutation_status = data[1:, 1:]
    mutation_status = mutation_status.astype(np.float)

    return mutation_names, patient_ID, mutation_status


def load_genetic_XNAT(genetic_info):
    """
    Load the patient IDs and genetic data from XNAT, Only works if you have a
    file /resources/GENETICS/files/genetics.json for each patient containing
    a single dictionary of all labels.

    Args:
        url (string): XNAT URL
        project: XNAT project ID

    Returns:
        mutation_names (numpy array): Names of the different genetic mutations
        patient_ID (numpy array): IDs of patients for which genetic data is
         loaded
        mutation_status (numpy array): The status of the different mutations
         for each patient
    """
    requests.packages.urllib3.disable_warnings(InsecureRequestWarning)

    config = load_config_XNAT(genetic_info)
    url = config['XNAT']['url']
    projectID = config['XNAT']['projectID']

    # Example
    # url = "http://bigr-rad-xnat.erasmusmc.nl"
    # projectID = 'LGG-Radiogenom'
    # url = genetic_info['url']
    # projectID = genetic_info['projectID']

    session = xnat.connect(url, verify=False)

    subkeys = session.projects[projectID].subjects.keys()
    subkeys.sort()
    session.disconnect()

    baseurl = url + '/data/archive/projects/' + projectID + '/subjects/'
    patient_ID = list()
    mutation_names = None
    mutation_status = list()
    for i_patient in subkeys:
        # Extra check as there are bound to be some fake patients
        if projectID in i_patient:
                patient_ID.append(i_patient)

                data = requests.get(baseurl + i_patient +
                                    '/resources/GENETICS/files/genetics.json')
                datadict = data.json()

                # Load and check the header
                if mutation_names is None:
                    mutation_names = list()
                    mutation_names_temp = datadict.keys()
                    for i_name in mutation_names_temp:
                        # convert u_str to str
                        mutation_names.append(str(i_name))

                mutation_status_temp = datadict.values()
                mutation_status.append(mutation_status_temp)

    mutation_names = np.asarray(mutation_names)
    mutation_status = np.asarray(mutation_status)

    return mutation_names, patient_ID, mutation_status


def findmutationdata(patientinfo, mutation_type, filenames,
                     image_features_temp=None):
    """
    Load the label data and match to the unage features.

    Args:
        patientinfo (string): file with patient label data
        mutation_type (string): name of the label read out from patientinfo
        filenames (list): names of the patient feature files, used for matching
        image_features (np.array or list): array of the features

    Returns:
        mutation_data (dict): contains patient ids, their labels and the mutation name
    """
    # Get the mutation labels and patient IDs
    mutation_data_temp = load_mutation_status(patientinfo, mutation_type)
    mutation_data = dict()
    patient_IDs = list()
    mutation_label = list()
    for i_len in range(len(mutation_data_temp['mutation_label'])):
        mutation_label.append(list())

    image_features = list()
    for i_feat, feat in enumerate(filenames):
        ifound = 0
        matches = list()
        for i_num, i_patient in enumerate(mutation_data_temp['patient_IDs']):
            if i_patient in str(feat):
                patient_IDs.append(i_patient)
                matches.append(i_patient)
                if image_features_temp is not None:
                    image_features.append(image_features_temp[i_feat])
                for i_len in range(len(mutation_data_temp['mutation_label'])):
                    mutation_label[i_len].append(mutation_data_temp['mutation_label'][i_len][i_num])
                ifound += 1

        if ifound > 1:
            message = ('Multiple matches ({}) found in labeling for feature file {}.').format(str(matches), str(feat))
            raise ae.PREDICTIOError(message)

        elif ifound == 0:
            message = ('No entry found in labeling for feature file {}.').format(str(feat))
            raise ae.PREDICTIOError(message)

    # if image_features_temp is not None:
    #     image_features = np.asarray(image_features)

    # Convert to arrays
    for i_len in range(len(mutation_label)):
        mutation_label[i_len] = np.asarray(mutation_label[i_len])

    mutation_data['patient_IDs'] = np.asarray(patient_IDs)
    mutation_data['mutation_label'] = np.asarray(mutation_label)
    mutation_data['mutation_name'] = mutation_data_temp['mutation_name']

    return mutation_data, image_features


def load_config_XNAT(config_file_path):
    '''
    Configparser for retreiving patient data from XNAT.
    '''
    settings = configparser.ConfigParser()
    settings.read(config_file_path)

    settings_dict = {'XNAT': dict()}

    settings_dict['XNAT']['url'] =\
        str(settings['Genetics']['url'])

    settings_dict['XNAT']['projectID'] =\
        str(settings['Genetics']['projectID'])

    return settings_dict
