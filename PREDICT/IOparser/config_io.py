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

import configparser
import numpy as np
import re


def load_config(config_file_path):
    """
    Load the config ini, parse settings to PREDICT

    Args:
        config_file_path (String): path of the .ini config file

    Returns:
        settings_dict (dict): dict with the loaded settings
    """

    settings = configparser.ConfigParser()
    settings.read(config_file_path)

    settings_dict = {'DataPaths': dict(), 'CrossValidation': dict(),
                     'Genetics': dict(), 'ImageFeatures': dict(),
                     'HyperOptimization': dict(), 'General': dict(),
                     'Classification': dict()}

    settings_dict['General']['cross_validation'] =\
        settings['General'].getboolean('cross_validation')

    settings_dict['General']['construction_type'] =\
        str(settings['General']['construction_type'])

    settings_dict['Classification']['classifier'] =\
        str(settings['Classification']['classifier'])

    # First load the datapaths
    settings_dict['DataPaths']['svm_file'] =\
        str(settings['DataPaths']['svm_file'])

    settings_dict['DataPaths']['genetic_file'] =\
        str(settings['DataPaths']['genetic_file'])

    settings_dict['DataPaths']['image_feature_file'] =\
        str(settings['DataPaths']['image_feature_file'])

    # Cross validation settings
    settings_dict['CrossValidation']['N_iterations'] =\
        settings['CrossValidation'].getint('N_iterations')

    settings_dict['CrossValidation']['test_size'] =\
        settings['CrossValidation'].getfloat('test_size')

    # Genetic settings
    mutation_setting = str(settings['Genetics']['mutation_type'])

    mutation_types = re.findall("\[(.*?)\]", mutation_setting)

    for i_index, i_mutation in enumerate(mutation_types):
        stripped_mutation_type = [x.strip() for x in i_mutation.split(',')]
        mutation_types[i_index] = stripped_mutation_type

    settings_dict['Genetics']['mutation_type'] =\
        mutation_types

    settings_dict['Genetics']['genetic_file'] =\
        str(settings['DataPaths']['genetic_file'])

    # Settings for image features
    settings_dict['ImageFeatures']['patient_root_folder'] =\
        str(settings['ImageFeatures']['patient_root_folder'])

    settings_dict['ImageFeatures']['image_folders'] =\
        [str(item).strip() for item in
         settings['ImageFeatures']['image_folders'].split(',')]

    settings_dict['ImageFeatures']['contour_files'] =\
        [str(item).strip() for item in
         settings['ImageFeatures']['contour_files'].split(',')]

    settings_dict['ImageFeatures']['image_type'] =\
        [str(item).strip() for item in
         settings['ImageFeatures']['image_type'].split(',')]

    settings_dict['ImageFeatures']['genetic_file'] =\
        str(settings['DataPaths']['genetic_file'])

    settings_dict['ImageFeatures']['image_feature_file'] =\
        str(settings['DataPaths']['image_feature_file'])

    # Gabor settings
    settings_dict['ImageFeatures']['gabor_settings'] = dict()

    gabor_frequencies = [str(item).strip() for item in
                         settings['ImageFeatures']['gabor_frequencies']
                         .split(',')]
    gabor_frequencies = np.asarray(gabor_frequencies).astype(np.float)

    gabor_angles = [str(item).strip() for item in
                    settings['ImageFeatures']['gabor_angles']
                    .split(',')]
    gabor_angles = np.asarray(gabor_angles).astype(np.float)
    # Convert gabor angle to radians from angles
    gabor_angles = np.radians(gabor_angles)

    settings_dict['ImageFeatures']['gabor_settings']['gabor_frequencies'] =\
        gabor_frequencies

    settings_dict['ImageFeatures']['gabor_settings']['gabor_angles'] =\
        gabor_angles

    settings_dict['HyperOptimization']['scoring_method'] =\
        str(settings['HyperOptimization']['scoring_method'])
    settings_dict['HyperOptimization']['test_size'] =\
        settings['HyperOptimization'].getfloat('test_size')
    settings_dict['HyperOptimization']['N_iter'] =\
        settings['HyperOptimization'].getint('N_iterations')
    settings_dict['HyperOptimization']['score_threshold'] =\
        settings['HyperOptimization'].getfloat('score_threshold')

    return settings_dict
