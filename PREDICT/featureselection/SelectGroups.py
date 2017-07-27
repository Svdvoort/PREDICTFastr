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

from sklearn.base import BaseEstimator
from sklearn.feature_selection.base import SelectorMixin
import numpy as np


class SelectGroups(BaseEstimator, SelectorMixin):
    def __init__(self, parameters):
        params = list()
        if parameters['histogram_features']:
            params.append('hf_')
        if parameters['shape_features']:
            params.append('sf_')
        if parameters['orientation_features']:
            params.append('of_')
        if parameters['semantic_features']:
            params.append('semf_')
        if parameters['patient_features']:
            params.append('pf_')
        if parameters['coliage_features']:
            params.append('cf_')
        if parameters['texture_features']:
            params.append('tf_')
        elif not parameters['texture_features']:
            pass
        else:
            params.append('tf_' + parameters['texture_features'])

        self.parameters = params

    def fit(self, feature_labels):
        '''
        Select only features specificed by parameters per patient
        '''
        # Remove NAN
        selectrows = list()
        for num, l in enumerate(feature_labels):
            if any(x in l for x in self.parameters):
                selectrows.append(num)

        self.selectrows = selectrows

    def transform(self, inputarray):
        return np.asarray([x[self.selectrows] for x in inputarray])

    def _get_support_mask(self):
        pass
