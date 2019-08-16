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


def get_patient_features(metadata, image_type, protocol_feat=False):
    patient_labels = list()
    patient_features = list()

    if [0x10, 0x1010] in list(metadata.keys()):
        try:
            patient_age = int(metadata[0x10, 0x1010].value[0:3])
        except ValueError:
            print("[PREDICT Warning] Patient age is not an integer, using zero.")
            patient_age = 0
    else:
        print("[PREDICT Warning] No patient age in metadata, using zero.")
        patient_age = 0

    patient_labels.append('pf_age')
    patient_features.append(patient_age)

    if [0x10, 0x40] in list(metadata.keys()):
        patient_sex = metadata[0x10, 0x40].value

        if patient_sex == 'M':
            patient_sex = 0
        elif patient_sex == 'F':
            patient_sex = 1
        else:
            patient_sex = 2
    else:
        print("[PREDICT Warning] No patient sex in metadata, using 2.")
        patient_sex = 2

    patient_labels.append('pf_sex')
    patient_features.append(patient_sex)

    # Include slice thickness
    if image_type == 'CT' or image_type == 'MR':
        if protocol_feat:
            slice_thickness = int(metadata[0x18, 0x50].value)
            patient_labels.append('pf_thickness')
            patient_features.append(slice_thickness)

            voxel_spacing = metadata[0x28, 0x30].value
            patient_labels.append('pf_voxelspacing0')
            patient_features.append(float(voxel_spacing[0]))
            patient_labels.append('pf_voxelspacing1')
            patient_features.append(float(voxel_spacing[1]))

    # Echo times and tesla for MR
    # if image_type == 'MR':
    #     TR = int(metadata[0x18, 0x80].value)
    #     patient_labels.append('pf_TR')
    #     patient_features.append(TR)
    #
    #     TE = int(metadata[0x18, 0x81].value)
    #     patient_labels.append('pf_TE')
    #     patient_features.append(TE)
    #
    #     Tesla = int(metadata[0x18, 0x87].value)
    #     patient_labels.append('pf_Tesla')
    #     patient_features.append(Tesla)

    return patient_features, patient_labels
