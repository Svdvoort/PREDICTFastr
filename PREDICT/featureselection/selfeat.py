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

from sklearn.feature_selection import VarianceThreshold


def selfeat_variance(image_features, labels, thresh=0.99):
    '''
    Select features using a variance threshold.
    '''
    sel = VarianceThreshold(threshold=thresh*(1 - thresh))
    sel = sel.fit(image_features)
    image_features = sel.transform(image_features)
    labels = sel.transform(labels).tolist()[0]

    return image_features, labels, sel
