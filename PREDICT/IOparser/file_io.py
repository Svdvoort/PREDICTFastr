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
import dicom as pydicom
import re
import pandas as pd
from natsort import natsorted, ns
import os

import SimpleITK as sitk
import PREDICT.helpers.sitk_helper as sitkh
import PREDICT.genetics.genetic_processing as gp
import PREDICT.addexceptions as PREDICTexceptions


def load_data(featurefiles, patientinfo=None, label_names=None, modnames=[]):
    ''' Read feature files and stack the features per patient in an array.
        Additionally, if a patient label file is supplied, the features from
        a patient will be matched to the labels.

        Parameters
        ----------
        featurefiles: list, mandatory
                List containing all paths to the .hdf5 feature files to be loaded.
                The argument should contain a list per modelity, e.g.
                [[features_mod1_patient1, features_mod1_patient2, ...],
                 [features_mod2_patient1, features_mod2_patient2, ...]].

        patientinfo: string, optional
                Path referring to the .txt file to be used to read patient
                labels from. See the Github Wiki for the format.

        label_names: list, optional
                List containing all the labels that should be extracted from
                the patientinfo file.

    '''

    # Read out all feature values and labels
    image_features_temp = list()
    feature_labels_all = list()
    for i_patient in range(0, len(featurefiles[0])):
        feature_values_temp = list()
        feature_labels_temp = list()
        for i_mod in range(0, len(featurefiles)):
            feat_temp = pd.read_hdf(featurefiles[i_mod][i_patient])
            feature_values_temp += feat_temp.feature_values
            if not modnames:
                # Create artificial names
                feature_labels_temp += [f + '_M' + str(i_mod) for f in feat_temp.feature_labels]
            else:
                # Use the provides modality names
                feature_labels_temp += [f + '_' + str(modnames[i_mod]) for f in feat_temp.feature_labels]

        image_features_temp.append((feature_values_temp, feature_labels_temp))

        # Also make a list of all unique label names
        feature_labels_all = feature_labels_all + list(set(feature_labels_temp) - set(feature_labels_all))

    # If some objects miss certain features, we will identify these with NaN values
    feature_labels_all.sort()
    image_features = list()
    for patient in image_features_temp:
        feat_temp = patient[0]
        label_temp = patient[1]

        feat = list()
        for f in feature_labels_all:
            if f in label_temp:
                index = label_temp.index(f)
                fv = feat_temp[index]
            else:
                fv = np.NaN
            feat.append(fv)

        image_features.append((feat, feature_labels_all))

    # Get the mutation labels and patient IDs
    if patientinfo is not None:
        # We use the feature files of the first modality to match to patient name
        pfiles = featurefiles[0]
        try:
            mutation_data, image_features =\
                gp.findmutationdata(patientinfo,
                                    label_names,
                                    pfiles,
                                    image_features)
        except ValueError as e:
            message = e.message + '. Please take a look at your labels' +\
                ' file and make sure it is formatted correctly. ' +\
                'See also https://github.com/MStarmans91/WORC/wiki/The-WORC-configuration#genetics.'
            raise PREDICTexceptions.PREDICTValueError(message)

        print("Mutation Labels:")
        print(mutation_data['mutation_label'])
        print('Total of ' + str(mutation_data['patient_IDs'].shape[0]) +
              ' patients')
        pos = np.sum(mutation_data['mutation_label'])
        neg = mutation_data['patient_IDs'].shape[0] - pos
        print(('{} positives, {} negatives').format(pos, neg))
    else:
        # Use filenames as patient ID s
        patient_IDs = list()
        for i in featurefiles:
            patient_IDs.append(os.path.basename(i))
        mutation_data = dict()
        mutation_data['patient_IDs'] = patient_IDs

    return mutation_data, image_features


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
