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
import scipy.stats
from radiomics import featureextractor
import PREDICT.addexceptions as ae
import PREDICT.helpers.image_helper as ih
import PREDICT.imagefeatures.histogram_features as hf


def gabor_filter_parallel(image, mask, parameters=dict(), n_jobs=1,
                          backend='threading'):
    """
    Apply gabor filters to image, done in parallel.
    Note: on a cluster, where parallelisation of the gabor filters
    is not possible, use backend="threading"
    """

    if "gabor_frequencies" in parameters.keys():
        gabor_frequencies = parameters["gabor_frequencies"]
    else:
        gabor_frequencies = [0.05, 0.2, 0.5]

    if "gabor_angles" in parameters.keys():
        gabor_angles = parameters["gabor_angles"]
    else:
        gabor_angles = [0, 45, 90, 135]

    # Create kernel from frequencies and angles
    kernels = list(itertools.product(gabor_frequencies,
                                     gabor_angles))

    N_slices = image.shape[2]
    N_kernels = len(kernels)

    # Filter the images with all kernels in paralell
    full_filtered = list()
    for i_slice in range(0, N_slices):
        print(('\t -- Filtering slice {} / {}.').format(str(i_slice + 1), N_slices))
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

    # Extract the features per kernel
    gabor_features = list()
    gabor_labels = list()
    for i_index, i_kernel in enumerate(kernels):
        # Get histogram features of Gabor image for full tumor
        histogram_features, histogram_labels = hf.get_histogram_features(full_filtered[i_index])
        histogram_labels = [l.replace('hf_', 'tf_Gabor_') for l in histogram_labels]
        gabor_features.extend(histogram_features)

        # Adjust feature labels
        i_kernel = [i_kernel[0], round(i_kernel[1], 2)]
        final_feature_names = [feature_name + '_F' + str(i_kernel[0]) + '_A' + str(i_kernel[1]) for feature_name in histogram_labels]
        gabor_labels.extend(final_feature_names)

    if len(gabor_features) != len(gabor_labels):
        raise ae.PREDICTValueError('Label length does not fit feature length')

    return gabor_features, gabor_labels


def gabor_filter(image, mask, kernel):
    '''
    Filter an image with a Gabor kernel. The kernel should be a list containing
    the frequency and the angle. After filtering, the image is flattened and
    masked by the mask.
    '''
    filtered_image, _ = skimage.filters.gabor(image,
                                              frequency=kernel[0],
                                              theta=kernel[1])
    filtered_image = filtered_image.flatten()
    mask = mask.flatten()
    filtered_image = filtered_image[mask]
    return filtered_image


def get_GLCM_features_multislice(image, mask, parameters=dict()):
    if "levels" in parameters.keys():
        levels = parameters["levels"]
    else:
        levels = 16

    if "angles" in parameters.keys():
        angles = parameters["angles"]
    else:
        angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]

    if "distances" in parameters.keys():
        distances = parameters["distances"]
    else:
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

        # Set all masked voxels to zero
        image_bounded[~mask_bounded] = 0

        # Rescale to reflect the number of levels required
        image_bounded = image_bounded + image_bounded.min()
        image_bounded = image_bounded*255.0 / image_bounded.max()

        image_bounded = image_bounded.astype(np.uint8)

        image_bounded = rescale_intensity(image_bounded, out_range=(0, levels-1))

        image_bounded = image_bounded.astype(np.uint8)

        # compute actual GLCM
        try:
            GLCM_matrix = greycomatrix(image_bounded, distances, angles,
                                       levels=levels, normed=True)

            contrast.append(greycoprops(GLCM_matrix, 'contrast').flatten())
            dissimilarity.append(greycoprops(GLCM_matrix, 'dissimilarity').flatten())
            homogeneity.append(greycoprops(GLCM_matrix, 'homogeneity').flatten())
            ASM.append(greycoprops(GLCM_matrix, 'ASM').flatten())
            energy.append(greycoprops(GLCM_matrix, 'energy').flatten())
            correlation.append(greycoprops(GLCM_matrix, 'correlation').flatten())

        except ValueError:
            print(f'[PREDICT WARNING] Slice {i_slice} to small to compute GLCM: {image_bounded.shape}.')

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

    GLCM_features = contrast_mean.tolist() + contrast_std.tolist() +\
        dissimilarity_mean.tolist() + dissimilarity_std.tolist() +\
        homogeneity_mean.tolist() + homogeneity_std.tolist() +\
        ASM_mean.tolist() + ASM_std.tolist() + energy_mean.tolist() +\
        energy_std.tolist() + correlation_mean.tolist() +\
        correlation_std.tolist()

    feature_names = ['tf_GLCMMS_contrast', 'tf_GLCMMS_dissimilarity',
                     'tf_GLCMMS_homogeneity', 'tf_GLCMMS_ASM',
                     'tf_GLCMMS_energy', 'tf_GLCMMS_correlation']

    GLCM_labels = list()
    for i_name, i_dist, i_angle in itertools.product(feature_names,
                                                     distances,
                                                     angles):
        # Round to reduce name length
        i_dist = round(i_dist, 2)
        i_angle = round(i_angle, 2)

        label_mean = i_name + 'd' + str(i_dist) + 'A' + str(i_angle) + 'mean'
        label_std = i_name + 'd' + str(i_dist) + 'A' + str(i_angle) + 'std'
        GLCM_labels.append(label_mean)
        GLCM_labels.append(label_std)

    if len(GLCM_features) != len(GLCM_labels):
        l1 = len(GLCM_features)
        l2 = len(GLCM_labels)
        raise ae.PREDICTValueError(f'Label length ({l1}) does not fit ' +
                                   f' feature length ({l2}).')

    return GLCM_features, GLCM_labels


def get_GLCM_features(image, mask, parameters=dict()):
    '''
    Compute Gray Level Co-occurence Matrix (GLCM) features. The image is first
    discretized to a set number of greyscale values. The GLCM will be computed
    at multiple distances and angles. The pixels outside the mask will always
    be set to zero.

    As the GLCM is defined in 2D, the GLCM for a 3D image will be computed
    by computing statistics over all the GLCM for all 2D axial slices, such
    as the mean and std.

    The output are two lists: the feature values and the labels.
    '''
    if "levels" in parameters.keys():
        levels = parameters["levels"]
    else:
        levels = 16

    if "angles" in parameters.keys():
        angles = parameters["angles"]
    else:
        angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]

    if "distances" in parameters.keys():
        distances = parameters["distances"]
    else:
        distances = [1, 3]

    N_slices = image.shape[2]

    contrast = list()
    dissimilarity = list()
    homogeneity = list()
    ASM = list()
    energy = list()
    correlation = list()

    GLCM_matrix = np.zeros([levels, levels, len(distances), len(angles)])
    for i_slice in range(0, N_slices):
        image_bounded, mask_bounded = bbox_2D(image[:, :, i_slice],
                                              mask[:, :, i_slice])

        # Set all masked voxels to zero
        image_bounded[~mask_bounded] = 0

        # Rescale to reflect the number of levels required
        image_bounded = image_bounded + image_bounded.min()
        image_bounded = image_bounded*255.0 / image_bounded.max()

        image_bounded = image_bounded.astype(np.uint8)

        image_bounded = rescale_intensity(image_bounded, out_range=(0, levels-1))

        image_bounded = image_bounded.astype(np.uint8)

        # compute actual GLCM
        try:
            GLCM_matrix += greycomatrix(image_bounded, distances, angles,
                                        levels=levels, normed=True)
        except ValueError:
            print(f'[PREDICT WARNING] Slice {i_slice} to small to compute GLCM: {image_bounded.shape}.')

    contrast = greycoprops(GLCM_matrix, 'contrast').flatten()
    dissimilarity = greycoprops(GLCM_matrix, 'dissimilarity').flatten()
    homogeneity = greycoprops(GLCM_matrix, 'homogeneity').flatten()
    ASM = greycoprops(GLCM_matrix, 'ASM').flatten()
    energy = greycoprops(GLCM_matrix, 'energy').flatten()
    correlation = greycoprops(GLCM_matrix, 'correlation').flatten()

    GLCM_features = contrast.tolist() +\
        dissimilarity.tolist() +\
        homogeneity.tolist() +\
        ASM.tolist() + energy.tolist() +\
        correlation.tolist()

    feature_names = ['tf_GLCM_contrast', 'tf_GLCM_dissimilarity',
                     'tf_GLCM_homogeneity', 'tf_GLCM_ASM', 'tf_GLCM_energy',
                     'tf_GLCM_correlation']

    GLCM_labels = list()
    for i_name, i_dist, i_angle in itertools.product(feature_names,
                                                     distances,
                                                     angles):
        # Round to reduce name length
        i_dist = round(i_dist, 2)
        i_angle = round(i_angle, 2)

        label = i_name + 'd' + str(i_dist) + 'A' + str(i_angle)
        GLCM_labels.append(label)

    if len(GLCM_features) != len(GLCM_labels):
        l1 = len(GLCM_features)
        l2 = len(GLCM_labels)
        raise ae.PREDICTValueError(f'Label length ({l1}) does not fit ' +
                                   f' feature length ({l2}).')

    return GLCM_features, GLCM_labels


def get_LBP_features(image, mask, parameters=dict()):
    '''
    Compute features by applying a Local Binary Pattern (LBP) filter to an image.
    The LBP will be constructed with a radius and neighboorhood defined by N_points.
    These must be provided as lists of integers.

    As LBP are defined in 2D, the LBP for a 3D image will be computed
    by computing statistics over all the LBP for all 2D axial slices, such
    as the mean and std.

    The output are two lists: the feature values and the labels.
    '''
    if "radius" in parameters.keys():
        radius = parameters["radius"]
    else:
        radius = [3, 8, 15]

    if "N_points" in parameters.keys():
        N_points = parameters["N_points"]
    else:
        N_points = [12, 24, 36]

    method = 'uniform'

    LBP_features = list()
    LBP_labels = list()

    mask = mask.flatten()

    for i_index, (i_radius, i_N_points) in enumerate(zip(radius, N_points)):
        LBP_image = np.zeros(image.shape)
        for i_slice in range(0, image.shape[2]):
            LBP_image[:, :, i_slice] = local_binary_pattern(image[:, :, i_slice], P=i_N_points, R=i_radius, method=method)

        # Extract histogram features from LBP image
        masked_voxels = ih.get_masked_voxels(LBP_image, mask)
        if masked_voxels.size == 0:
            print("[PREDICT Warning] LBP features, fully empty. Using zeros.")
            masked_voxels = [0]

        feature_values, feature_names = hf.get_histogram_features(masked_voxels)

        # Alter labels and add labels and values to respective lists
        feature_names = [l.replace('hf_', 'tf_LBP_') for l in feature_names]
        LBP_features.extend(feature_values)
        final_feature_names = [feature_name + '_R' + str(i_radius) + '_P' + str(i_N_points) for feature_name in feature_names]
        LBP_labels.extend(final_feature_names)

    return LBP_features, LBP_labels


def get_GLSZM_features(image, mask):
    mask = mask.astype(int)
    image = sitk.GetImageFromArray(image)
    mask = sitk.GetImageFromArray(mask)
    kwargs = {'binWidth': 25,
              'interpolator': sitk.sitkBSpline,
              'resampledPixelSpacing': None,
              'verbose': True}

    # Initialize wrapperClass to generate signature
    success = False
    while not success:
        try:
            extractor = featureextractor.RadiomicsFeatureExtractor(**kwargs)
            extractor.disableAllFeatures()
            extractor.enableFeatureClassByName('glszm')
            featureVector = extractor.execute(image, mask)
            success = True
        except RuntimeError as e:
            print(f'[PREDICT WARNING] {e} : doubling bin width.')
            kwargs['binWidth'] = 2 * kwargs['binWidth']

    # Assign features to corresponding groups
    GLSZM_labels_temp = list()
    GLSZM_features = list()

    for featureName in featureVector.keys():
        # Skip the "general" features
        if 'glszm' in featureName:
            GLSZM_labels_temp.append(featureName)
            GLSZM_features.append(featureVector[featureName])

    # Replace part of label to indicate a texture feature
    GLSZM_labels = list()
    for l in GLSZM_labels_temp:
        l = l.replace('original_glszm', 'tf_GLSZM')
        GLSZM_labels.append(l)

    return GLSZM_features, GLSZM_labels


def get_GLRLM_features(image, mask):
    mask = mask.astype(int)
    image = sitk.GetImageFromArray(image)
    mask = sitk.GetImageFromArray(mask)
    kwargs = {'binWidth': 25,
              'interpolator': sitk.sitkBSpline,
              'resampledPixelSpacing': None,
              'verbose': True}

    # Initialize wrapperClass to generate signature
    extractor = featureextractor.RadiomicsFeatureExtractor(**kwargs)
    extractor.disableAllFeatures()
    extractor.enableFeatureClassByName('glrlm')

    featureVector = extractor.execute(image, mask)

    # Assign features to corresponding groups
    GLRLM_labels_temp = list()
    GLRLM_features = list()

    for featureName in featureVector.keys():
        # Skip the "general" features
        if 'glrlm' in featureName:
            GLRLM_labels_temp.append(featureName)
            GLRLM_features.append(featureVector[featureName])

    # Replace part of label to indicate a texture feature
    GLRLM_labels = list()
    for l in GLRLM_labels_temp:
        l = l.replace('original_glrlm', 'tf_GLRLM')
        GLRLM_labels.append(l)

    return GLRLM_features, GLRLM_labels


def get_NGTDM_features(image, mask):
    mask = mask.astype(int)
    image = sitk.GetImageFromArray(image)
    mask = sitk.GetImageFromArray(mask)
    kwargs = {'binWidth': 25,
              'interpolator': sitk.sitkBSpline,
              'resampledPixelSpacing': None,
              'verbose': True}

    # Initialize wrapperClass to generate signature
    extractor = featureextractor.RadiomicsFeatureExtractor(**kwargs)
    extractor.disableAllFeatures()
    extractor.enableFeatureClassByName('ngtdm')
    extractor.settings['distances'] = [1]

    featureVector = extractor.execute(image, mask)

    # Assign features to corresponding groups
    NGTDM_labels_temp = list()
    NGTDM_features = list()

    for featureName in featureVector.keys():
        # Skip the "general" features
        if 'ngtdm' in featureName:
            NGTDM_labels_temp.append(featureName)
            NGTDM_features.append(featureVector[featureName])

    # Replace part of label to indicate a texture feature
    NGTDM_labels = list()
    for l in NGTDM_labels_temp:
        l = l.replace('original_ngtdm', 'tf_NGTDM')
        NGTDM_labels.append(l)

    return NGTDM_features, NGTDM_labels


def get_texture_features(image, mask, parameters=None, config=None,
                         config_general=None):
    if parameters is None:
        parameters = dict()
        parameters['gabor_settings'] = dict()
        parameters['LBP'] = dict()
        parameters['GLCM'] = dict()

    # Check whether specific parameters for using joblib are given
    if 'Joblib_ncores' in config_general.keys():
        n_jobs = config_general['Joblib_ncores']
    else:
        n_jobs = None

    if 'Joblib_backend' in config_general.keys():
        backend = config_general['Joblib_backend']
    else:
        backend = None

    texture_features = []
    texture_labels = []
    if config is None:
        print('Computing all texture features.')
        print("\t Computing GLCM features.")
        GLCM_features, GLCM_labels = get_GLCM_features(image, mask,
                                                       parameters['GLCM'])
        GLCMMS_features, GLCMMS_labels =\
            get_GLCM_features_multislice(image, mask, parameters['GLCM'])

        print("\t Computing GLRLM features.")
        GLRLM_features, GLRLM_labels = get_GLRLM_features(image, mask)

        print("\t Computing GLSZM features.")
        GLSZM_features, GLSZM_labels = get_GLSZM_features(image, mask)

        print("\t Computing LBP features.")
        LBP_features, LBP_labels = get_LBP_features(image, mask,
                                                    parameters['LBP'])

        print("\t Computing NGTDM features.")
        NGTDM_features, NGTDM_labels = get_NGTDM_features(image, mask)

        print("\t Computing Gabor features.")
        gabor_features, gabor_labels =\
            gabor_filter_parallel(image, mask, parameters['gabor_settings'],
                                  n_jobs=n_jobs, backend=backend)

        texture_features = gabor_features + GLCM_features + GLCMMS_features +\
            GLRLM_features + GLSZM_features + NGTDM_features + LBP_features

        texture_labels = gabor_labels + GLCM_labels + GLCMMS_labels +\
            GLRLM_labels + GLSZM_labels + NGTDM_labels + LBP_labels

    if config['texture_LBP']:
        print("\t Computing LBP features.")
        texture_features_tmp, texture_labels_tmp = \
            get_LBP_features(image, mask, parameters['LBP'])

        texture_features += texture_features_tmp
        texture_labels += texture_labels_tmp

    if config['texture_GLCM']:
        print("\t Computing GLCM features.")
        texture_features_tmp, texture_labels_tmp =\
            get_GLCM_features(image, mask, parameters['GLCM'])

        texture_features += texture_features_tmp
        texture_labels += texture_labels_tmp

    if config['texture_GLCMMS']:
        print("\t Computing GLCMMS features.")
        texture_features_tmp, texture_labels_tmp =\
            get_GLCM_features_multislice(image, mask, parameters['GLCM'])

        texture_features += texture_features_tmp
        texture_labels += texture_labels_tmp

    if config['texture_GLRLM']:
        print("\t Computing GLRLM features.")
        texture_features_tmp, texture_labels_tmp =\
            get_GLRLM_features(image, mask)

        texture_features += texture_features_tmp
        texture_labels += texture_labels_tmp

    if config['texture_GLSZM']:
        print("\t Computing GLSZM features.")
        texture_features_tmp, texture_labels_tmp =\
            get_GLSZM_features(image, mask)

        texture_features += texture_features_tmp
        texture_labels += texture_labels_tmp

    if config['texture_NGTDM']:
        print("\t Computing NGTDM features.")
        texture_features_tmp, texture_labels_tmp =\
            get_NGTDM_features(image, mask)

        texture_features += texture_features_tmp
        texture_labels += texture_labels_tmp

    if config['texture_Gabor']:
        print("\t Computing Gabor features.")
        texture_features_tmp, texture_labels_tmp =\
            gabor_filter_parallel(image, mask,  parameters['gabor_settings'],
                                  n_jobs=n_jobs, backend=backend)

        texture_features += texture_features_tmp
        texture_labels += texture_labels_tmp

    return texture_features, texture_labels


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
