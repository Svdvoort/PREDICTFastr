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

import pandas as pd

import PREDICT.helpers.sitk_helper as sitkh
import PREDICT.helpers.image_helper as ih

import PREDICT.imagefeatures.histogram_features as hf
import PREDICT.imagefeatures.texture_features as tf
import PREDICT.imagefeatures.shape_features as sf
import PREDICT.imagefeatures.semantic_features as semf
import PREDICT.imagefeatures.orientation_features as of
import PREDICT.imagefeatures.coliage_features as cf
import PREDICT.imagefeatures.dti_features as dtif
import PREDICT.imagefeatures.patient_features as pf
import PREDICT.imagefeatures.log_features as logf
import PREDICT.imagefeatures.vessel_features as vesf
import PREDICT.imagefeatures.phase_features as phasef
import PREDICT.addexceptions as ae


# CONSTANTS
N_BINS = 50


def get_image_features(image_data, mask, parameters,
                       config=None,
                       config_general=None,
                       output=None):
    '''
    Calculate features from a ROI of an image.

    Parameters
    ----------
    image_data: Pandas DataFrame or dictionary, mandatory
            Contains the image, image type, metadata and semantics for
            the feature extraction. These have to be indexed by these keys.

            Should either be a Pandas DataFrame, in which the image type is
            he key of the images, or a dictionary, in which image type is a
            a separate field.

            The image should be a SimpleITK Image, the image type a string,
            the metadata a pydicom dicom type and the semantics a dictionary.


    mask: ITK Image, mandatory
            ROI to be used for feature extraction.

    parameters: dictionary, mandatory,
            Parameters for feature calculation. See the Github Wiki for the possible
            fields and their description.

    config: dictionary, mandatory
            Configuration for feature calculation. Mostly configures which
            features are calculated or not. See the Github Wiki for the possible
            fields and their description.

    config_general: dictionary, mandatory
            Configuration for general settings. Currently only configures
            settings for the Joblib Parallel function. See the Github Wiki
            for the possible fields and their description.

    output: string, mandatory
            path referring to the .hdf5 file to which the output should be
            written for the CalcFeatures function. This field is used to match
            the patient ID of the semantic features to this filename.


    Returns
    ----------
    feature_values: list
            Contains the values for all extracted features.

    feature_labels: list
            Contains the labels for all extracted features. Each entry
            corresponds to the element with the same index from the
            feature_values object.

    '''

    # Assign data to correct variables
    if type(image_data) == pd.core.frame.DataFrame:
        image_type = image_data['images'].keys()[0]
        meta_data = image_data['metadata']
        sem_data = image_data['semantics']
        image_data = image_data['images']
    else:
        # Dictionary
        image_type = image_data['image_type']
        meta_data = image_data['metadata']
        sem_data = image_data['semantics']
        image_data = image_data['images']

    # Initialize feature value and label lists.
    feature_values = list()
    feature_labels = list()

    # Extract shape features
    if len(mask.GetSize()) == 3:
        shape_mask = ih.get_masked_slices_mask(mask)
    else:
        shape_mask = mask

    if config["shape"]:
        print("\t Computing shape features.")
        shape_features, shape_labels = sf.get_shape_features(shape_mask,
                                                             meta_data)
        feature_values += shape_features
        feature_labels += shape_labels

    if config["orientation"]:
        print("\t Computing orientation features.")
        orientation_features, orientation_labels =\
            of.get_orientation_features(shape_mask)
        feature_values += orientation_features
        feature_labels += orientation_labels

    if meta_data is not None:
        print("\t Extracting patient features.")
        patient_features, patient_labels =\
            pf.get_patient_features(meta_data, image_type)
        feature_values += patient_features
        feature_labels += patient_labels

    if sem_data is not None and output is not None:
        print("\t Extracting semantic features.")
        sem_features, sem_labels = semf.get_semantic_features(sem_data,
                                                              output)
        feature_values += sem_features
        feature_labels += sem_labels

    if 'DTI_post' in image_type:
        dti_features, dti_labels = dtif.get_dti_post_features(image_data,
                                                              mask,
                                                              meta_data)
        feature_values += dti_features
        feature_labels += dti_labels

    elif 'DTI' in image_type:
        dti_features, dti_labels = dtif.get_dti_features(image_data, mask,
                                                         meta_data)
        feature_values += dti_features
        feature_labels += dti_labels

    elif any(type in image_type for type in ['MR', 'CT', 'PET', 'MG']):
        image_data_array = sitkh.GetArrayFromImage(image_data)
        mask_array = sitkh.GetArrayFromImage(mask)

        masked_voxels = ih.get_masked_voxels(image_data_array, mask_array)

        if config["histogram"]:
            print("\t Computing histogram features.")
            histogram_features, histogram_labels =\
                hf.get_histogram_features(masked_voxels,
                                          N_BINS)

            feature_values += histogram_features
            feature_labels += histogram_labels

        # NOTE: As a minimum of 4 voxels in each dimension is needed
        # for the SimpleITK log filter, we compute these features
        # on the full image
        if config["log"]:
            print("\t Computing log features.")
            log_features, log_labels =\
                logf.get_log_features(image_data_array,
                                      mask_array,
                                      parameters['log'])
            feature_values += log_features
            feature_labels += log_labels

        image_data_array, mask_array = ih.get_masked_slices_image(
            image_data_array, mask_array)

        texture_features, texture_labels =\
            tf.get_texture_features(image_data_array,
                                    mask_array,
                                    parameters,
                                    config,
                                    config_general)

        feature_values += texture_features
        feature_labels += texture_labels

        if config["coliage"]:
            print("\t Computing coliage features.")
            coliage_features, coliage_labels =\
                cf.get_coliage_features(image_data_array,
                                        mask_array)
            feature_values += coliage_features
            feature_labels += coliage_labels

        if config["vessel"]:
            print("\t Computing vessel features.")
            vessel_features, vessel_labels =\
                vesf.get_vessel_features(image_data_array,
                                         mask_array,
                                         parameters['vessel'])
            feature_values += vessel_features
            feature_labels += vessel_labels

        if config["phase"]:
            print("\t Computing phase features.")
            phase_features, phase_labels =\
                phasef.get_phase_features(image_data_array,
                                          mask_array,
                                          parameters['phase'])
            feature_values += phase_features
            feature_labels += phase_labels

    else:
        raise ae.PREDICTTypeError(("Invalid image type: {}").format(image_type))

    return feature_values, feature_labels
