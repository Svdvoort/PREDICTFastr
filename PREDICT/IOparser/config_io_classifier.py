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
                     'Classification': dict(), 'SelectFeatGroup': dict(),
                     'Featsel': dict(), 'FeatureScaling': dict(),
                     'SampleProcessing': dict(), 'Imputation': dict(),
                     'Ensemble': dict()}

    settings_dict['General']['cross_validation'] =\
        settings['General'].getboolean('cross_validation')

    settings_dict['General']['Joblib_ncores'] =\
        settings['PREDICTGeneral'].getint('Joblib_ncores')

    settings_dict['General']['Joblib_backend'] =\
        str(settings['PREDICTGeneral']['Joblib_backend'])

    settings_dict['General']['tempsave'] =\
        settings['PREDICTGeneral'].getboolean('tempsave')

    settings_dict['Featsel']['Variance'] =\
        [str(item).strip() for item in
         settings['Featsel']['Variance'].split(',')]

    settings_dict['Featsel']['SelectFromModel'] =\
        [str(item).strip() for item in
         settings['Featsel']['SelectFromModel'].split(',')]

    settings_dict['Featsel']['UsePCA'] =\
        [str(item).strip() for item in
         settings['Featsel']['UsePCA'].split(',')]

    settings_dict['Featsel']['PCAType'] =\
        [str(item).strip() for item in
         settings['Featsel']['PCAType'].split(',')]

    settings_dict['Featsel']['StatisticalTestUse'] =\
        [str(item).strip() for item in
         settings['Featsel']['StatisticalTestUse'].split(',')]

    settings_dict['Featsel']['StatisticalTestMetric'] =\
        [str(item).strip() for item in
         settings['Featsel']['StatisticalTestMetric'].split(',')]

    settings_dict['Featsel']['StatisticalTestThreshold'] =\
        [float(str(item).strip()) for item in
         settings['Featsel']['StatisticalTestThreshold'].split(',')]

    settings_dict['Featsel']['ReliefUse'] =\
        [str(item).strip() for item in
         settings['Featsel']['ReliefUse'].split(',')]

    settings_dict['Featsel']['ReliefNN'] =\
        [int(str(item).strip()) for item in
         settings['Featsel']['ReliefNN'].split(',')]

    settings_dict['Featsel']['ReliefSampleSize'] =\
        [int(str(item).strip()) for item in
         settings['Featsel']['ReliefSampleSize'].split(',')]

    settings_dict['Featsel']['ReliefDistanceP'] =\
        [int(str(item).strip()) for item in
         settings['Featsel']['ReliefDistanceP'].split(',')]

    settings_dict['Featsel']['ReliefNumFeatures'] =\
        [int(str(item).strip()) for item in
         settings['Featsel']['ReliefNumFeatures'].split(',')]

    settings_dict['Imputation']['use'] =\
        [boolean(str(item).strip()) for item in
         settings['Imputation']['use'].split(',')]

    settings_dict['Imputation']['strategy'] =\
        [str(item).strip() for item in
         settings['Imputation']['strategy'].split(',')]

    settings_dict['Imputation']['n_neighbors'] =\
        [int(str(item).strip()) for item in
         settings['Imputation']['n_neighbors'].split(',')]

    settings_dict['General']['FeatureCalculator'] =\
        str(settings['General']['FeatureCalculator'])

    # Feature selection options
    for key in settings['SelectFeatGroup'].keys():
        settings_dict['SelectFeatGroup'][key] =\
            [str(item).strip() for item in
             settings['SelectFeatGroup'][key].split(',')]

    # Classification options
    settings_dict['Classification']['fastr'] =\
        settings['Classification'].getboolean('fastr')

    settings_dict['Classification']['fastr_plugin'] =\
        str(settings['Classification']['fastr_plugin'])

    settings_dict['Classification']['classifier'] =\
        str(settings['Classification']['classifier'])

    settings_dict['Classification']['Kernel'] =\
        str(settings['Classification']['Kernel'])

    # Cross validation settings
    settings_dict['CrossValidation']['N_iterations'] =\
        settings['CrossValidation'].getint('N_iterations')

    settings_dict['CrossValidation']['test_size'] =\
        settings['CrossValidation'].getfloat('test_size')

    # Genetic settings
    settings_dict['Genetics']['label_names'] =\
        [str(item).strip() for item in
         settings['Genetics']['label_names'].split(',')]

    settings_dict['Genetics']['modus'] =\
        str(settings['Genetics']['modus'])

    # Settings for hyper optimization
    settings_dict['HyperOptimization']['scoring_method'] =\
        str(settings['HyperOptimization']['scoring_method'])
    settings_dict['HyperOptimization']['test_size'] =\
        settings['HyperOptimization'].getfloat('test_size')
    settings_dict['HyperOptimization']['N_iter'] =\
        settings['HyperOptimization'].getint('N_iterations')
    settings_dict['HyperOptimization']['n_jobspercore'] =\
        int(settings['HyperOptimization']['n_jobspercore'])

    settings_dict['FeatureScaling']['scale_features'] =\
        settings['FeatureScaling'].getboolean('scale_features')
    settings_dict['FeatureScaling']['scaling_method'] =\
        str(settings['FeatureScaling']['scaling_method'])

    settings_dict['SampleProcessing']['SMOTE'] =\
        [boolean(str(item).strip()) for item in
         settings['SampleProcessing']['SMOTE'].split(',')]

    settings_dict['SampleProcessing']['SMOTE_ratio'] =\
        [float(str(item).strip()) for item in
         settings['SampleProcessing']['SMOTE_ratio'].split(',')]

    settings_dict['SampleProcessing']['SMOTE_neighbors'] =\
        [int(str(item).strip()) for item in
         settings['SampleProcessing']['SMOTE_ratio'].split(',')]

    settings_dict['SampleProcessing']['Oversampling'] =\
        [boolean(str(item).strip()) for item in
         settings['SampleProcessing']['Oversampling'].split(',')]

    settings_dict['Ensemble']['Use'] =\
        settings['Ensemble'].getboolean('Use')

    return settings_dict
