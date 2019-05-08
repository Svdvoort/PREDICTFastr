#!/usr/bin/env python

# Copyright 2011-2018 Biomedical Imaging Group Rotterdam, Departments of
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
import glob
import pydicom
import re
import pandas as pd
from natsort import natsorted, ns
import os

import SimpleITK as sitk
import PREDICT.helpers.sitk_helper as sitkh
import PREDICT.addexceptions as PREDICTexceptions


def load_dicom(dicom_folder):
    dicom_reader = sitk.ImageSeriesReader()
    dicom_file_names = dicom_reader.GetGDCMSeriesFileNames(dicom_folder)
    dicom_reader.SetFileNames(dicom_file_names)
    dicom_image = dicom_reader.Execute()

    image_metadata = pydicom.read_file(dicom_file_names[0])

    return dicom_image, image_metadata


def get_b_value(dicom_file):
    # Of course there is no standard on how to the b-value of DTI is stored
    # So different ways of reading it, based on manufacturer
    manufacturer = dicom_file[0x8, 0x70].value
    if 'GE' in manufacturer.upper():
        # GE stores it under 'Slop_int_6' in tag 0043,1039
        # Sometimes stored as string, sometimes as array, handle differences
        b_tag = dicom_file[0x43, 0x1039].value
        if isinstance(b_tag, basestring):
            matchObj = re.match('(\d+)', b_tag)
            b_value = float(matchObj.group(1))
        elif isinstance(b_tag, list):
            b_value = b_tag[0]
    else:
        print('Unknown manufacturer!' + manufacturer)
        b_value = None
    return b_value


def get_gradient(dicom_file):
    # Of course there is no standard on how to the gradient of DTI is stored
    # So different ways of reading it, based on manufacturer
    manufacturer = dicom_file[0x8, 0x70].value
    if 'GE' in manufacturer.upper():
        # GE stores it under 'Slop_int_6' in tag 0043,1039
        # Sometimes stored as string, sometimes as array, handle differences
        x_gradient = dicom_file[0x19, 0x10bb].value
        y_gradient = dicom_file[0x19, 0x10bc].value
        z_gradient = dicom_file[0x19, 0x10bd].value
        gradient_data = {'x_gradient': float(x_gradient),
                         'y_gradient': float(y_gradient),
                         'z_gradient': float(z_gradient)}
    else:
        print('Unknown manufacturer!' + manufacturer)
        gradient_data = None
    return gradient_data


def load_DTI(dicom_folder):
    # DTI are also dicom, but loaded a bit differently
    # Unfortunately we can't use SimpleITK to read private tags, so need
    # pydicom.....

    dicom_images = get_DTI_dicoms(dicom_folder)
    positions = get_positions(dicom_images)

    # Find the unique position to know where new series starts
    unique_positions = np.squeeze(np.argwhere(np.all(positions == positions[0],
                                                     axis=1)))

    unique_images = list()
    b_values = list()
    gradient_data = list()
    for i_index, i_position in enumerate(unique_positions):
        # Get b_value from first dicom in sequence
        b_values.append(get_b_value(dicom_images[i_position]))
        gradient_data.append(get_gradient(dicom_images[i_position]))

        if i_index < len(unique_positions) - 1:
            temp_image_data = np.asarray([o.pixel_array for o in
                                         dicom_images[i_position:
                                          unique_positions[i_index+1]]])
        else:  # At the end of the array
            temp_image_data = np.asarray([o.pixel_array for o in
                                         dicom_images[i_position:]])

        # Transpose to get in same frame as sitk images
        temp_image_data = np.transpose(temp_image_data, [2, 1, 0])
        # Save it as ITK image, to keep everything in same format
        temp_image_data = sitkh.GetImageFromArray(temp_image_data)
        # temp_image_data = sitk.Normalize(temp_image_data)
        unique_images.append(temp_image_data)

    meta_data = {'b_values': b_values, 'dicom_meta_data': dicom_images[0],
                 'gradient_data': gradient_data}

    return unique_images, meta_data


def load_DTI_post(dicom_folder, patient_ID):
    # Going to load the 3 eigenvalues
    L1_file = os.path.join(dicom_folder, patient_ID + '_DTI_post_L1.nii.gz')
    L2_file = os.path.join(dicom_folder, patient_ID + '_DTI_post_L2.nii.gz')
    L3_file = os.path.join(dicom_folder, patient_ID + '_DTI_post_L3.nii.gz')

    L1_image = sitk.ReadImage(L1_file)
    L2_image = sitk.ReadImage(L2_file)
    L3_image = sitk.ReadImage(L3_file)

    DTI_post_images = [L1_image, L2_image, L3_image]

    return DTI_post_images, None

def get_DTI_dicoms(input_dir):

    # We need to sort them in a certain way, otherwise 10 will come before 100.
    # This will screw up the order of the images, fixed in this way.
    dicom_files = glob.glob(input_dir+'/*.dcm')
    dicom_files = natsorted(dicom_files, alg=ns.IGNORECASE)

    dicoms = list()
    for i_dicom in dicom_files:
        dicoms.append(pydicom.read_file(i_dicom))

    return dicoms


def get_positions(dicoms):

    positions = list()
    for i_file, i_dicom in enumerate(dicoms):
        positions.append(i_dicom.ImagePositionPatient)

    positions = np.asarray(positions)
    return positions


def load_image_features(image_feature_file, patient_ID, genetic_file,
                        image_folders, image_type, contour_files,
                        gabor_settings):

    panda_data_loaded = pd.read_hdf(image_feature_file)

    same_genetic = panda_data.genetic_file is genetic_file
    same_image = panda_data.image_folders is image_folders
    same_image_type = panda_data.image_type is image_type
    same_contour = panda_data.contour_files is contour_files
    same_gabor_settings = panda_data.gabor_settings is gabor_settings

    same_overall = same_genetic and same_image and same_image_type and \
        same_contour and same_gabor_settings
    if same_overall:
        image_features = save_shelve['image_features']
    else:
        image_features = None
    save_shelve.close()

    return image_features, same_overall
