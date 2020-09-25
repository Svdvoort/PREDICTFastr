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


def get_dicom_features(metadata, image_type, tags, labels):
    """Extract values from DICOM tags as features."""
    # Initialize objects
    dicom_labels = list()
    dicom_features = list()

    # Check which tags are in DICOM
    keys = [str(i) for i in metadata.keys()]
    values = list(metadata.values())

    for tag, label in zip(tags, labels):
        # Convert tag to something matching DICOM keys
        tag = '(' + tag + ')'

        if tag == '(0010, 1010)':
            # Get patient age
            value = get_patient_age(metadata)

        elif tag == '(0010, 0040)':
            # Get patient sex
            value = get_patient_sex(metadata)

        elif tag == '(0018, 0087)':
            # Get magnetic field strength
            value = get_magnetic_field_strength(metadata)

        elif tag == '(0028, 0030)':
            # Get pixel spacing
            value = get_pixel_spacing(metadata)

        elif tag == '(0018, 0022)' and label == 'FatSat':
            # Get whether scan has fat saturation or not
            value = get_fatsat(metadata)

        elif tag == '(0008, 0070)':
            # Get scanner manufacturer
            value = get_manufacturer(metadata)

        else:
            # Undefined preset, simply extract the value
            if tag in keys:
                try:
                    value = float(values[keys.index(tag)].value)
                except ValueError:
                    print(f"[PREDICT Warning] Value of {tag} is not a float, using NaN.")
                    value = np.NaN
            else:
                print(f"[PREDICT Warning] {tag} not in metadata, using NaN.")
                value = np.NaN

        dicom_labels.append('df_' + label)
        dicom_features.append(value)

    return dicom_features, dicom_labels


def get_patient_age(metadata):
    """Extract patient age from DICOM tags."""
    if [0x10, 0x1010] in list(metadata.keys()):
        try:
            patient_age = float(metadata[0x10, 0x1010].value[0:3])
        except ValueError:
            print("[PREDICT Warning] Patient age is not a float, using NaN.")
            patient_age = np.NaN
    else:
        print("[PREDICT Warning] No patient age in metadata, using NaN.")
        patient_age = np.NaN

    return patient_age


def get_patient_sex(metadata):
    """Extract patient sex from DICOM tags."""
    if [0x10, 0x40] in list(metadata.keys()):
        patient_sex = metadata[0x10, 0x40].value

        if patient_sex == 'M':
            patient_sex = 0.0
        elif patient_sex == 'F':
            patient_sex = 1.0
        else:
            patient_sex = 2.0
    else:
        print("[PREDICT Warning] No patient sex in metadata, using NaN.")
        patient_sex = np.NaN

    return patient_sex


def get_magnetic_field_strength(metadata):
    """Extract magnetic field strength from DICOM tags."""
    if [0x18, 0x87] in list(metadata.keys()):
        try:
            tesla = metadata[0x18, 0x87].value
            tesla = float(tesla)
        except ValueError:
            print(f"[PREDICT Warning] Magnetic field strenght is not a float ({tesla}), using NaN.")
            tesla = np.NaN

        # Make some corrections
        if tesla == 0.5 or tesla == 5000.0:
            tesla = 0.5
        elif tesla == 1.0 or tesla == 10000.0:
            tesla = 1.0
        elif tesla == 1.5 or tesla == 15000.0:
            tesla = 1.5
        elif tesla == 3.0 or tesla == 30000.0:
            tesla = 3.0

    else:
        print("[PREDICT Warning] Magnetic field strength not in metadata, using NaN.")
        tesla = np.NaN

    return tesla


def get_pixel_spacing(metadata):
    """Extract pixel spacing from DICOM tags.

    Assume x- and y-spacing is the same, so just use the x-spacing.
    """
    if [0x28, 0x30] in list(metadata.keys()):
        try:
            pixel_spacing = metadata[0x28, 0x30].value
            pixel_spacing = float(pixel_spacing[0])
        except ValueError:
            print(f"[PREDICT Warning] Pixel spacing is not a float ({pixel_spacing}), using NaN.")
            pixel_spacing = np.NaN

    else:
        print("[PREDICT Warning] No pixel spacing in metadata, using NaN.")
        pixel_spacing = np.NaN

    return pixel_spacing


def get_fatsat(metadata):
    """Extract whether scan has fat saturation from DICOM tags."""
    if [0x18, 0x22] in list(metadata.keys()):
        fatsat = metadata[0x18, 0x22].value

        if type(fatsat) == int:
            fatsat = np.NaN
        else:
            if 'FS' in fatsat or 'SFS' in fatsat:
                fatsat = 1.0
            else:
                fatsat = 0.0

    else:
        print("[PREDICT Warning] No fat saturation in metadata, using NaN.")
        fatsat = np.NaN

    return fatsat


def get_manufacturer(metadata):
    """Extract scanner manufacturer from DICOM tags."""
    if [0x8, 0x70] in list(metadata.keys()):
        manufacturer = metadata[0x8, 0x70].value

        if 'Siemens' in manufacturer or 'SIEMENS' in manufacturer:
            manufacturer = 0.0
        elif 'Philips' in manufacturer or 'PHILIPS' in manufacturer:
            manufacturer = 1.0
        elif 'GE' in manufacturer:
            manufacturer = 2.0
        elif 'Toshiba' in manufacturer or 'TOSHIBA' in manufacturer:
            manufacturer = 3.0
        else:
            print(f"[PREDICT Warning] Manufacturer {manufacturer} unknown, using NaN.")
            manufacturer = np.NaN

    else:
        print("[PREDICT Warning] No manufacturer in metadata, using NaN.")
        manufacturer = np.NaN

    return manufacturer
