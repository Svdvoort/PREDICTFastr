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
import scipy.stats

import PREDICT.helpers.sitk_helper as sitkh


def get_dti_features(image, mask, meta_data):
    i_mask_array = sitkh.GetArrayFromImage(mask)
    i_mask_array = i_mask_array.astype(np.bool)
    i_mask_array = i_mask_array.flatten()

    array_list = list()

    for i_dti in image:
        i_dti = sitkh.GetArrayFromImage(i_dti).flatten()
        i_dti = i_dti[i_mask_array]
        array_list.append(i_dti)

    i_image_array = np.vstack(array_list)

    # Adjust such that everything is positive, needed for log
    i_image_array = i_image_array + np.min(i_image_array) + 1

    b_values = meta_data['b_values']
    A = np.vstack([b_values, np.ones(len(b_values))]).T
    # Do least squares fit and get the slope, which is the ADC
    # This will give us the ADC in mm^2/s
    ADC_tumor_voxels = -np.linalg.lstsq(A, np.log(i_image_array))[0][0, :]

    ADC_mean = np.mean(ADC_tumor_voxels)
    ADC_std = np.std(ADC_tumor_voxels)
    ADC_min = np.min(ADC_tumor_voxels)
    ADC_max = np.max(ADC_tumor_voxels)

    dti_features = [ADC_mean, ADC_std, ADC_min, ADC_max]
    dti_labels = ['ADC_mean', 'ADC_std', 'ADC_min', 'ADC_max']

    return dti_features, dti_labels


def get_dti_post_features(image, mask, meta_data):
    i_mask_array = sitkh.GetArrayFromImage(mask)
    i_mask_array = i_mask_array.astype(np.bool)
    i_mask_array = i_mask_array.flatten()

    L1_image = sitkh.GetArrayFromImage(image[0]).flatten()
    L2_image = sitkh.GetArrayFromImage(image[1]).flatten()
    L3_image = sitkh.GetArrayFromImage(image[2]).flatten()

    L1_image = L1_image[i_mask_array]
    L2_image = L2_image[i_mask_array]
    L3_image = L3_image[i_mask_array]

    # A small fix because sometimes the eigenvalues are just below/around 0
    # And will give error in division.

    if np.amin(L1_image) <= 0:
        L1_image = L1_image + np.abs(np.amin(L1_image)) + 0.00001
    if np.amin(L2_image) <= 0:
        L2_image = L2_image + np.abs(np.amin(L2_image)) + 0.00001
    if np.amin(L3_image) <= 0:
        L3_image = L3_image + np.abs(np.amin(L3_image)) + 0.00001

    ADC_map = (L1_image + L2_image + L3_image)/3.0

    # ADC map is also mean diffusivity
    FA_numerator = 3.0*((L1_image - ADC_map)**2.0 + (L2_image - ADC_map)**2.0 + (L3_image - ADC_map)**2.0)
    FA_denominator = 2.0*(L1_image**2.0 + L2_image**2.0 + L3_image**2.0)
    FA_map = np.sqrt(FA_numerator/FA_denominator)

    # print(min(L2_image))
    # print(min(L3_image))

    # Also volume ratio
    VR_map = 1.0 - L1_image*L2_image*L3_image/ADC_map**3.0

    # Now get inside tumor

    # ADC_tumor = ADC_map[i_mask_array]
    # FA_tumor = FA_map[i_mask_array]
    # VR_tumor = VR_map[i_mask_array]

    ADC_min, ADC_max, ADC_mean, ADC_std, ADC_median, ADC_skew, ADC_kurtosis, ADC_range =\
        get_statistical_moments(ADC_map)

    FA_min, FA_max, FA_mean, FA_std, FA_median, FA_skew, FA_kurtosis, FA_range =\
        get_statistical_moments(FA_map)

    VR_min, VR_max, VR_mean, VR_std, VR_median, VR_skew, VR_kurtosis, VR_range =\
        get_statistical_moments(VR_map)

    # dti_features = [ADC_min, ADC_max, ADC_mean, ADC_std, ADC_median, ADC_skew, ADC_kurtosis, ADC_range,
    #                 FA_min, FA_max, FA_mean, FA_std, FA_median, FA_skew, FA_kurtosis, FA_range,
    #                 VR_min, VR_max, VR_mean, VR_std, VR_median, VR_skew, VR_kurtosis, VR_range]
    #
    # panda_labels = ['ADC_min', 'ADC_max', 'ADC_mean', 'ADC_std', 'ADC_median', 'ADC_skew', 'ADC_kurtosis', 'ADC_range',
    #                 'FA_min', 'FA_max', 'FA_mean', 'FA_std', 'FA_median', 'FA_skew', 'FA_kurtosis', 'FA_range',
    #                 'VR_min', 'VR_max', 'VR_mean', 'VR_std', 'VR_median', 'VR_skew', 'VR_kurtosis', 'VR_range']

    dti_features = [ADC_min, ADC_max, ADC_mean, ADC_std, ADC_median, ADC_skew, ADC_kurtosis, ADC_range]

    dti_labels = ['ADC_min', 'ADC_max', 'ADC_mean', 'ADC_std', 'ADC_median', 'ADC_skew', 'ADC_kurtosis', 'ADC_range']

    return dti_features, dti_labels


def get_statistical_moments(tumor_map):

    min_val = np.percentile(tumor_map, 2)
    max_val = np.percentile(tumor_map, 98)
    mean_val = np.mean(tumor_map)
    std_val = np.std(tumor_map)
    median_val = np.std(tumor_map)
    skew_val = scipy.stats.skew(tumor_map)
    kurtosis_val = scipy.stats.kurtosis(tumor_map)
    range_val = max_val - min_val

    return min_val, max_val, mean_val, std_val, median_val, skew_val, kurtosis_val, range_val
