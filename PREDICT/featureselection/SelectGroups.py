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

from sklearn.base import BaseEstimator
from sklearn.feature_selection.base import SelectorMixin
import numpy as np


class SelectGroups(BaseEstimator, SelectorMixin):
    '''
    Object to fit feature selection based on the type group the feature belongs
    to. The label for the feature is used for this procedure.
    '''
    def __init__(self, parameters):
        '''
        Parameters
        ----------
        parameters: dict, mandatory
                Contains the settings for the groups to be selected. Should
                contain the settings for the following groups:
                - histogram_features
                - shape_features
                - orientation_features
                - semantic_features
                - patient_features
                - coliage_features
                - phase_features
                - vessel_features
                - log_features
                - texture_Gabor_features
                - texture_GLCM_features
                - texture_GLCMMS_features
                - texture_GLRLM_features
                - texture_GLSZM_features
                - texture_NGTDM_features
                - texture_LBP_features

        '''
        params = list()
        if parameters['histogram_features'] == 'True':
            params.append('hf_')
        if parameters['shape_features'] == 'True':
            params.append('sf_')
        if parameters['orientation_features'] == 'True':
            params.append('of_')
        if parameters['semantic_features'] == 'True':
            params.append('semf_')
        if parameters['patient_features'] == 'True':
            params.append('pf_')
        if parameters['coliage_features'] == 'True':
            params.append('cf_')
        if parameters['phase_features'] == 'True':
            params.append('phasef_')
        if parameters['vessel_features'] == 'True':
            params.append('vf_')
        if parameters['log_features'] == 'True':
            params.append('logf_')

        if 'texture_features' in parameters.keys():
            # Backwards compatability
            if parameters['texture_features'] == 'True':
                params.append('tf_')
            elif parameters['texture_features'] == 'False':
                pass
            else:
                params.append('tf_' + parameters['texture_features'])
        else:
            # Hyperparameter per feature group
            if parameters['texture_gabor_features'] == 'True':
                params.append('tf_Gabor')
            if parameters['texture_glcm_features'] == 'True':
                params.append('tf_GLCM_')
            if parameters['texture_glcmms_features'] == 'True':
                params.append('tf_GLCMMS')
            if parameters['texture_glrlm_features'] == 'True':
                params.append('tf_GLRLM')
            if parameters['texture_glszm_features'] == 'True':
                params.append('tf_GLSZM')
            if parameters['texture_ngtdm_features'] == 'True':
                params.append('tf_NGTDM')
            if parameters['texture_lbp_features'] == 'True':
                params.append('tf_LBP')

        self.parameters = params

    def fit(self, feature_labels):
        '''
        Select only features specificed by parameters per patient.

        Parameters
        ----------
        feature_labels: list, optional
                Contains the labels of all features used. The index in this
                list will be used in the transform funtion to select features.
        '''
        # Remove NAN
        selectrows = list()
        for num, l in enumerate(feature_labels):
            if any(x in l for x in self.parameters):
                selectrows.append(num)

        self.selectrows = selectrows

    def transform(self, inputarray):
        '''
        Transform the inputarray to select only the features based on the
        result from the fit function.

        Parameters
        ----------
        inputarray: numpy array, mandatory
                Array containing the items to use selection on. The type of
                item in this list does not matter, e.g. floats, strings etc.
        '''
        return np.asarray([np.asarray(x)[self.selectrows].tolist() for x in inputarray])

    def _get_support_mask(self):
        # NOTE: Method is required for the Selector class, but can be empty
        pass
