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

import pandas as pd


def get_patient_features(metadata, image_type):
    panda_labels = list()
    patient_features = list()

    patient_age = int(metadata[0x10, 0x1010].value[0:3])
    panda_labels.append('patient_age')
    patient_features.append(patient_age)

    patient_sex = metadata[0x10, 0x40].value

    if patient_sex == 'M':
        patient_sex = 0
    elif patient_sex == 'F':
        patient_sex = 1
    else:
        patient_sex = 2

    panda_labels.append('patient_sex')
    patient_features.append(patient_sex)

    if image_type == 'CT':
        slice_thickness = int(metadata[0x18, 0x50].value)
        # TODO: create classifications for different kernels
        # kernel = int(metadata[0x18, 0x1210].value)
        panda_labels.append('slice_thickness')
        patient_features.append(slice_thickness)

    panda_dict = dict(zip(panda_labels, patient_features))

    patient_dict = dict()
    patient_dict['all'] = pd.Series(panda_dict)

    patient_features = pd.Series(patient_dict)

    return patient_features
