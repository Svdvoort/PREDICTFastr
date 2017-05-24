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
import skimage.filters
from joblib import Parallel, delayed
import itertools
from skimage.feature import greycomatrix, greycoprops
from skimage.exposure import rescale_intensity
from skimage.feature import local_binary_pattern
import SimpleITK as sitk

import pandas as pd

import scipy.stats
import PREDICT.IOparser.config_general as config_io

from radiomics import featureextractor


def gabor_filter_parallel(image, mask, gabor_settings, n_jobs=None,
                          backend=None):
    """
    Apply gabor filters to image, done in parallel.
    Note: on a cluster, where parallelisation of the gabor filters
    is not possible, use backend="threading"
    """

    config = config_io.load_config()
    if n_jobs is None:
        n_jobs = config['Joblib']['njobs']
    if backend is None:
        backend = config['Joblib']['backend']

    # Create kernel from frequencies and angles
    kernels = list(itertools.product(gabor_settings['gabor_frequencies'],
                                     gabor_settings['gabor_angles']))

    N_slices = image.shape[2]
    N_kernels = len(kernels)
    print(N_slices)

    features = np.zeros([N_kernels, 2, N_slices])
    full_filtered = list()
    for i_slice in range(0, N_slices):
        filtered = Parallel(n_jobs=n_jobs, backend=backend)(delayed(gabor_filter)
                                                            (image=image[:, :, i_slice],
                                                            mask=mask[:, :, i_slice],
                                                            kernel=kernel)
                                                            for kernel in
                                                            kernels)
        # filtered_image.append(filtered)
        for i_index, i_kernel in enumerate(kernels):
            # features[i_index, 0, i_slice] = filtered[i_index].mean()
            # features[i_index, 1, i_slice] = filtered[i_index].var()
            if i_slice == 0:
                full_filtered.append(filtered[i_index])
            else:
                full_filtered[i_index] = np.append(full_filtered[i_index], filtered[i_index])

    mean_gabor = list()
    std_gabor = list()
    min_gabor = list()
    max_gabor = list()
    skew_gabor = list()
    kurt_gabor = list()
    for i_index, i_kernel in enumerate(kernels):
        mean_gabor.append(np.mean(full_filtered[i_index]))
        std_gabor.append(np.std(full_filtered[i_index]))
        min_gabor.append(np.percentile(full_filtered[i_index], 2))
        max_gabor.append(np.percentile(full_filtered[i_index], 98))
        skew_gabor.append(scipy.stats.skew(full_filtered[i_index]))
        kurt_gabor.append(scipy.stats.kurtosis(full_filtered[i_index]))
    # features = np.mean(features, 2).flatten()
    # features = features.tolist()

    features = mean_gabor + std_gabor + min_gabor + max_gabor + skew_gabor + kurt_gabor

    # Create labels
    panda_labels = list()
    for i_kernel in kernels:
        label_mean = 'f' + str(i_kernel[0]) + 'A' + str(i_kernel[1]) + 'mean'
        label_std = 'f' + str(i_kernel[0]) + 'A' + str(i_kernel[1]) + 'std'
        label_min = 'f' + str(i_kernel[0]) + 'A' + str(i_kernel[1]) + 'min'
        label_max = 'f' + str(i_kernel[0]) + 'A' + str(i_kernel[1]) + 'max'
        label_skew = 'f' + str(i_kernel[0]) + 'A' + str(i_kernel[1]) + 'skew'
        label_kurt = 'f' + str(i_kernel[0]) + 'A' + str(i_kernel[1]) + 'kurt'
        panda_labels.append(label_mean)
        panda_labels.append(label_std)
        panda_labels.append(label_min)
        panda_labels.append(label_max)
        panda_labels.append(label_skew)
        panda_labels.append(label_kurt)

    if len(features) != len(panda_labels):
        raise ValueError('Label length does not fit feature length')

    gabor_features = dict(zip(panda_labels, features))

    # Construct pandas series of features
    # gabor_features = pd.Series(panda_dict)

    return gabor_features


def gabor_filter(image, mask, kernel):
    filtered_image, _ = skimage.filters.gabor(image,
                                              frequency=kernel[0],
                                              theta=kernel[1])
    filtered_image = filtered_image.flatten()
    mask = mask.flatten()
    filtered_image = filtered_image[mask]
    return filtered_image


def get_GLCM_features(image, mask):

    angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]
    distances = [1, 3]

    N_slices = image.shape[2]

    contrast = list()
    dissimilarity = list()
    homogeneity = list()
    ASM = list()
    energy = list()
    correlation = list()

    for i_slice in range(0, N_slices):
        image_bounded, mask_bounded = bbox_2D(image[:, :, i_slice],
                                              mask[:, :, i_slice])

        image_bounded[~mask_bounded] = 0
        image_bounded = image_bounded + image_bounded.min()
        image_bounded = image_bounded*255.0 / image_bounded.max()

        image_bounded = image_bounded.astype(np.uint8)

        image_bounded = rescale_intensity(image_bounded, out_range=(0, 15))

        GLCM_matrix = greycomatrix(image_bounded, distances, angles, levels=16,
                                   normed=True)

        contrast.append(greycoprops(GLCM_matrix, 'contrast').flatten())
        dissimilarity.append(greycoprops(GLCM_matrix, 'dissimilarity').flatten())
        homogeneity.append(greycoprops(GLCM_matrix, 'homogeneity').flatten())
        ASM.append(greycoprops(GLCM_matrix, 'ASM').flatten())
        energy.append(greycoprops(GLCM_matrix, 'energy').flatten())
        correlation.append(greycoprops(GLCM_matrix, 'correlation').flatten())

    contrast_mean = np.mean(contrast, 0)
    contrast_std = np.std(contrast, 0)

    dissimilarity_mean = np.mean(dissimilarity, 0)
    dissimilarity_std = np.std(dissimilarity, 0)

    homogeneity_mean = np.mean(homogeneity, 0)
    homogeneity_std = np.std(homogeneity, 0)

    ASM_mean = np.mean(ASM, 0)
    ASM_std = np.std(ASM, 0)

    energy_mean = np.mean(energy, 0)
    energy_std = np.std(energy, 0)

    correlation_mean = np.mean(correlation, 0)
    correlation_std = np.std(correlation, 0)

    features = contrast_mean.tolist() + contrast_std.tolist() + dissimilarity_mean.tolist() +\
                dissimilarity_std.tolist() + homogeneity_mean.tolist() + homogeneity_std.tolist() + ASM_mean.tolist() +\
                ASM_std.tolist() + energy_mean.tolist() + energy_std.tolist() + correlation_mean.tolist() +\
                correlation_std.tolist()

    feature_names = ['contrast','dissimilarity', 'homogeneity', 'ASM', 'energy', 'correlation']

    panda_labels = list()
    for i_name, i_dist, i_angle in itertools.product(feature_names, distances, angles):
        label_mean = i_name + 'd' + str(i_dist) + 'A' + str(i_angle) + 'mean'
        label_std = i_name + 'd' + str(i_dist) + 'A' + str(i_angle) + 'std'
        panda_labels.append(label_mean)
        panda_labels.append(label_std)

    if len(features) != len(panda_labels):
        print(len(features))
        print(len(panda_labels))
        raise ValueError('Label length does not fit feature length')

    GCLM_features = dict(zip(panda_labels, features))

    return GCLM_features


def get_LBP_features(image, mask):

    LBP_image = np.zeros(image.shape)
    for i_slice in range(0, image.shape[2]):
        LBP_image[:, :, i_slice] = local_binary_pattern(image[:, :, i_slice], P=24, R=3, method='uniform')

    LBP_image = LBP_image.flatten()
    mask = mask.flatten()
    LBP_tumor = LBP_image[mask]

    min_val = np.percentile(LBP_tumor, 2)
    max_val = np.percentile(LBP_tumor, 98)
    mean_val = np.mean(LBP_tumor)
    std_val = np.std(LBP_tumor)
    median_val = np.median(LBP_tumor)
    skew_val = scipy.stats.skew(LBP_tumor)
    kurtosis_val = scipy.stats.kurtosis(LBP_tumor)
    range_val = max_val - min_val

    features = [min_val, max_val, mean_val, std_val, median_val, skew_val,
                kurtosis_val, range_val]

    feature_labels = ['LBP_min', 'LBP_max', 'LBP_std', 'LBP_median', 'LBP_skew',
                      'LBP_kurtosis', 'LBP_range']

    features = dict(zip(feature_labels, features))

    return features


def get_GLSZM_features(image, mask):
    mask = mask.astype(int)
    image = sitk.GetImageFromArray(image)
    mask = sitk.GetImageFromArray(mask)
    kwargs = {'binWidth': 25,
              'interpolator': sitk.sitkBSpline,
              'resampledPixelSpacing': None,
              'verbose': True}

    # Initialize wrapperClass to generate signature
    extractor = featureextractor.RadiomicsFeaturesExtractor(**kwargs)
    extractor.disableAllFeatures()
    extractor.enableFeatureClassByName('glszm')

    featureVector = extractor.execute(image, mask)

    # Assign features to corresponding groups
    GLSZM_labels = list()
    GLSZM_features = list()

    for featureName in featureVector.keys():
        # Skip the "general" features
        if 'glszm' in featureName:
            GLSZM_labels.append(featureName)
            GLSZM_features.append(featureVector[featureName])

    features = dict(zip(GLSZM_labels, GLSZM_features))

    return features


def get_GLRLM_features(image, mask):
    mask = mask.astype(int)
    image = sitk.GetImageFromArray(image)
    mask = sitk.GetImageFromArray(mask)
    kwargs = {'binWidth': 25,
              'interpolator': sitk.sitkBSpline,
              'resampledPixelSpacing': None,
              'verbose': True}

    # Initialize wrapperClass to generate signature
    extractor = featureextractor.RadiomicsFeaturesExtractor(**kwargs)
    extractor.disableAllFeatures()
    extractor.enableFeatureClassByName('glrlm')

    featureVector = extractor.execute(image, mask)

    # Assign features to corresponding groups
    GLRLM_labels = list()
    GLRLM_features = list()

    for featureName in featureVector.keys():
        # Skip the "general" features
        if 'glrlm' in featureName:
            GLRLM_labels.append(featureName)
            GLRLM_features.append(featureVector[featureName])

    features = dict(zip(GLRLM_labels, GLRLM_features))

    return features


def get_texture_features(image, mask, gabor_settings=None, config='LBP'):

    texture_features = dict()
    if config == 'all':
        gabor_features = gabor_filter_parallel(image, mask, gabor_settings)
        GLCM_features = get_GLCM_features(image, mask)
        GLRLM_features = get_GLRLM_features(image, mask)
        GLSZM_features = get_GLSZM_features(image, mask)
        LBP_features = get_LBP_features(image, mask)

        # texture_features = gabor_features.copy()
        # texture_features.update(GLCM_features)
        # texture_features.update(LBP_features)

        texture_features['Gabor'] = pd.Series(gabor_features)
        texture_features['GLCM'] = pd.Series(GLCM_features)
        texture_features['GLRLM'] = pd.Series(GLRLM_features)
        texture_features['GLSZM'] = pd.Series(GLSZM_features)
        texture_features['LBP'] = pd.Series(LBP_features)

        texture_features = pd.Series(texture_features)

    elif config == 'LBP':
        texture_features['LBP'] = pd.Series(get_LBP_features(image, mask))
        texture_features = pd.Series(texture_features)

    elif config == 'GLCM':
        texture_features['GLCM'] = pd.Series(get_GLCM_features(image, mask))
        texture_features = pd.Series(texture_features)

    elif config == 'GLRLM':
        texture_features['GLRLM'] = pd.Series(get_GLRLM_features(image, mask))
        texture_features = pd.Series(texture_features)

    elif config == 'GLSZM':
        texture_features['GLSZM'] = pd.Series(get_GLSZM_features(image, mask))
        texture_features = pd.Series(texture_features)

    elif config == 'Gabor':
        texture_features['Gabor'] = pd.Series(gabor_filter_parallel(image, mask, gabor_settings))
        texture_features = pd.Series(texture_features)

    return texture_features


def bbox_2D(img, mask):
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)

    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]

    img = img[rmin:rmax+1, cmin:cmax+1]
    mask = mask[rmin:rmax+1, cmin:cmax+1]
    return img, mask


def bbox_3D(img, mask):
    r = np.any(mask, axis=(1, 2))
    c = np.any(mask, axis=(0, 2))
    z = np.any(mask, axis=(0, 1))

    rmin, rmax = np.where(r)[0][[0, -1]]
    cmin, cmax = np.where(c)[0][[0, -1]]
    zmin, zmax = np.where(z)[0][[0, -1]]

    img = img[rmin:rmax+1, cmin:cmax+1, zmin:zmax+1]
    mask = mask[rmin:rmax+1, cmin:cmax+1, zmin:zmax+1]

    return img, mask
