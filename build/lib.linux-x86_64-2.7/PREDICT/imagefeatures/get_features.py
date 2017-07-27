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

import PREDICT.helpers.sitk_helper as sitkh
import PREDICT.imagefeatures.histogram_features as hf
import PREDICT.imagefeatures.texture_features as tf
import PREDICT.imagefeatures.shape_features as sf
import PREDICT.imagefeatures.semantic_features as semf
import PREDICT.imagefeatures.orientation_features as of
import PREDICT.imagefeatures.coliage_features as cf
import PREDICT.imagefeatures.dti_features as dtif
import PREDICT.imagefeatures.patient_features as pf
import PREDICT.helpers.image_helper as ih

# CONSTANTS
N_BINS = 50


def get_image_features(image_data, masks, gabor_settings,
                       mask_index=-1, multi_mask=False, config=None,
                       output=None):

    if type(image_data) == pd.core.frame.DataFrame:
        images = image_data['images']
        image_types = image_data['images'].keys()
        meta_data = image_data['metadata']
        sem_data = image_data['semantics']
        N_images = len(image_data.images)
    else:
        # Dictionary
        images = image_data['images']
        image_types = image_data['image_type']
        meta_data = image_data['metadata']
        sem_data = image_data['semantics']
        N_images = len(images)

    N_masks = len(masks)
    feature_values = list()
    feature_labels = list()

    if ~multi_mask and mask_index == -1:
        raise ValueError('Multi_mask was set to False, but no mask index was\
                         provided')

    if multi_mask and N_images != N_masks:
        raise ValueError('Multi_contour was set to True, but the number of\
                         contours does not match the number of images')

    if multi_mask:
        pass
    else:
        shape_mask = ih.get_masked_slices_mask(masks[mask_index])

        print("Computing shape features.")
        shape_features, shape_labels = sf.get_shape_features(shape_mask,
                                                             meta_data[0])
        feature_values += shape_features
        feature_labels += shape_labels

        if config["orientation"]:
            print("Computing orientation features.")
            orientation_features, orientation_labels =\
                of.get_orientation_features(shape_mask)
            feature_values += orientation_features
            feature_labels += orientation_labels

    if meta_data[0] is not None:
        print("Extracting patient features.")
        patient_features, patient_labels =\
            pf.get_patient_features(meta_data[0], image_types[0])
        feature_values += patient_features
        feature_labels += patient_labels

    if sem_data[0] is not None and output is not None:
        print("Extracting semantic features.")
        sem_features, sem_labels = semf.get_semantic_features(sem_data[0],
                                                              output)
        feature_values += sem_features
        feature_labels += sem_labels

    for i_image, i_mask, i_image_type, i_meta_data in zip(images, masks,
                                                          image_types,
                                                          meta_data):
        if 'MR' in i_image_type:
            i_image_array = sitkh.GetArrayFromImage(i_image)
            i_mask_array = sitkh.GetArrayFromImage(i_mask)

            masked_voxels = ih.get_masked_voxels(i_image_array, i_mask_array)

            print("Computing histogram features.")
            histogram_features, histogram_labels =\
                hf.get_histogram_features(masked_voxels,
                                          N_BINS)

            i_image_array, i_mask_array = ih.get_masked_slices_image(
                i_image_array, i_mask_array)

            print("Computing texture features.")
            texture_features, texture_labels =\
                tf.get_texture_features(i_image_array,
                                        i_mask_array,
                                        gabor_settings,
                                        config['texture'])

            feature_values += histogram_features + texture_features
            feature_labels += histogram_labels + texture_labels

            if config["coliage"]:
                print("Computing coliage features.")
                coliage_features, coliage_labels =\
                    cf.get_coliage_features(i_image_array,
                                            i_mask_array)
                feature_values += coliage_features
                feature_labels += coliage_labels

        elif 'DTI_post' in i_image_type:
            dti_features, dti_labels = dtif.get_dti_post_features(i_image,
                                                                  i_mask,
                                                                  i_meta_data)
            feature_values += dti_features
            feature_labels += dti_labels

        elif 'DTI' in i_image_type:
            dti_features, dti_labels = dtif.get_dti_features(i_image, i_mask,
                                                             i_meta_data)
            feature_values += dti_features
            feature_labels += dti_labels

        elif 'CT' in i_image_type:
            i_image_array = sitkh.GetArrayFromImage(i_image)
            i_mask_array = sitkh.GetArrayFromImage(i_mask)

            masked_voxels = ih.get_masked_voxels(i_image_array, i_mask_array)

            print("Computing histogram features.")
            histogram_features, histogram_labels =\
                hf.get_histogram_features(masked_voxels,
                                          N_BINS)

            i_image_array, i_mask_array = ih.get_masked_slices_image(
                i_image_array, i_mask_array)

            print("Computing texture features.")
            texture_features, texture_labels =\
                tf.get_texture_features(i_image_array,
                                        i_mask_array,
                                        gabor_settings,
                                        config['texture'])

            feature_values += histogram_features + texture_features
            feature_labels += histogram_labels + texture_labels

            if config["coliage"]:
                print("Computing coliage features.")
                coliage_features, coliage_labels =\
                    cf.get_coliage_features(i_image_array,
                                            i_mask_array)
                feature_values += coliage_features
                feature_labels += coliage_labels

        else:
            print("Invalid image type: {}").format(i_image_type)
            raise TypeError

    return feature_values, feature_labels
