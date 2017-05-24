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

    settings_dict = {'General': dict(), 'CrossValidation': dict(),
                     'Genetics': dict(), 'HyperOptimization': dict(),
                     'Classification': dict()}

    settings_dict['General']['cross_validation'] =\
        settings['General'].getboolean('cross_validation')

    settings_dict['General']['construction_type'] =\
        str(settings['General']['construction_type'])

    settings_dict['General']['gridsearch'] =\
        str(settings['General']['gridsearch_SVM'])

    settings_dict['General']['fastr'] =\
        settings['General'].getboolean('fastr')

    settings_dict['Classification']['fastr'] =\
        settings['Classification'].getboolean('fastr')

    settings_dict['Classification']['classifier'] =\
        str(settings['Classification']['classifier'])

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

    # Settings for hyper optimization
    settings_dict['HyperOptimization']['scoring_method'] =\
        str(settings['HyperOptimization']['scoring_method'])
    settings_dict['HyperOptimization']['test_size'] =\
        settings['HyperOptimization'].getfloat('test_size')
    settings_dict['HyperOptimization']['N_iter'] =\
        settings['HyperOptimization'].getint('N_iterations')
    settings_dict['HyperOptimization']['score_threshold'] =\
        settings['HyperOptimization'].getfloat('score_threshold')

    return settings_dict
