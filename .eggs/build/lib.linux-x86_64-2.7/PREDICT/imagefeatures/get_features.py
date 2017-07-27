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
    image_features = dict()

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

        shape_features = sf.get_shape_features(shape_mask, meta_data[0])

        if config["orientation"]:
            orientation_features = of.get_orientation_features(shape_mask)
            image_features['orientation_features'] = orientation_features

    image_features['shape_features'] = shape_features

    if meta_data[0] is not None:
        patient_features = pf.get_patient_features(meta_data[0],
                                                   image_types[0])
        image_features['patient_features'] = patient_features

    if sem_data[0] is not None and output is not None:
        sem_features = semf.get_semantic_features(sem_data[0], output)
        image_features['semantic_features'] = sem_features

    for i_image, i_mask, i_image_type, i_meta_data in zip(images, masks,
                                                          image_types,
                                                          meta_data):
        if 'MR' in i_image_type:
            i_image_array = sitkh.GetArrayFromImage(i_image)
            i_mask_array = sitkh.GetArrayFromImage(i_mask)

            masked_voxels = ih.get_masked_voxels(i_image_array, i_mask_array)

            histogram_features = hf.get_histogram_features(masked_voxels,
                                                           N_BINS)

            i_image_array, i_mask_array = ih.get_masked_slices_image(
                i_image_array, i_mask_array)

            texture_features = tf.get_texture_features(i_image_array,
                                                       i_mask_array,
                                                       gabor_settings,
                                                       config['texture'])

            coliage_features = cf.get_coliage_featues(i_image_array,
                                                      i_mask_array)
            # filter_features = ff.get_filter_features(i_image_array,
            #                                          i_mask_array)

            # image_features[i_image_type] =\
            #     pd.concat([histogram_features, texture_features],
            #               keys=['histogram_features', 'texture_features'])
            image_features['histogram_features'] = histogram_features
            image_features['texture_features'] = texture_features
            image_features['coliage_features'] = coliage_features
            # image_features['filter_features'] = filter_features
            # image_features[i_image_type] = histogram_features

        elif 'DTI_post' in i_image_type:
            dti_features = dtif.get_dti_post_features(i_image, i_mask, i_meta_data)
            image_features[i_image_type] = dti_features
        elif 'DTI' in i_image_type:
            dti_features = dtif.get_dti_features(i_image, i_mask, i_meta_data)
            image_features[i_image_type] = dti_features
        elif 'CT' in i_image_type:
            i_image_array = sitkh.GetArrayFromImage(i_image)
            i_mask_array = sitkh.GetArrayFromImage(i_mask)

            masked_voxels = ih.get_masked_voxels(i_image_array, i_mask_array)

            histogram_features = hf.get_histogram_features(masked_voxels,
                                                           N_BINS)

            i_image_array, i_mask_array = ih.get_masked_slices_image(
                i_image_array, i_mask_array)

            texture_features = tf.get_texture_features(i_image_array,
                                                       i_mask_array,
                                                       gabor_settings,
                                                       config['texture'])

            coliage_features = cf.get_coliage_featues(i_image_array,
                                                      i_mask_array)

            # filter_features = ff.get_filter_features(i_image_array,
            #                                          i_mask_array)

            # image_features[i_image_type] = histogram_features
            image_features['histogram_features'] = histogram_features
            image_features['texture_features'] = texture_features
            image_features['coliage_features'] = coliage_features
            # image_features['filter_features'] = filter_features
        else:
            print("Invalid image type: {}").format(i_image_type)
            raise TypeError

    # We also return just the arrray
    image_feature_array = list()

    for _, feattype in image_features.iteritems():
        for _, features in feattype.iteritems():
            image_feature_array.extend(features.values)

    print image_feature_array

    image_feature_array = np.asarray(image_feature_array)
    image_feature_array = image_feature_array.ravel()

    return image_features, image_feature_array
