#!/usr/bin/env python

# Copyright 2017-2018 Biomedical Imaging Group Rotterdam, Departments of
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

import PREDICT.helpers.contour_functions as cf


def get_histogram_features(data, N_bins=50):
    '''
    Compute histogram or first order features.

    Parameters
    ----------
    data: numpy array, mandatory
            1D array from which the features are extracted.

    N_bins: integer, mandatory
            Number of bins to be used in histogram creation.

    Returns
    ----------
    histogram_features: list
            Contains the values for all extracted features.

    histogram_labels: list
            Contains the labels for all extracted features. Each entry
            corresponds to the element with the same index from the
            histogram_features object.

    '''
    # Features computed on raw data because histogram creation loses
    # neccesary data:
    hist_min = get_min(data)
    hist_max = get_max(data)
    hist_range = get_range(data)
    quartile_range = get_quartile_range(data)

    # Features computed on histogram/discretized image to be more robust to outliers
    temp_histogram, temp_bins = create_histogram(data, N_bins)
    discretized_image = list()
    for d, b in zip(temp_histogram, temp_bins):
        if d != 0:
            discretized_image.extend([b] * d)

    discretized_image = np.asarray(discretized_image)
    hist_std = get_std(discretized_image)
    hist_skewness = get_skewness(discretized_image)
    hist_kurtosis = get_kurtosis(discretized_image)
    hist_peak = get_peak(temp_histogram, temp_bins)
    hist_peak_position = get_peak_position(temp_histogram)
    energy = get_energy(discretized_image)
    entropy = get_entropy(discretized_image)
    hist_mean = get_mean(discretized_image)
    hist_median = get_median(discretized_image)

    histogram_labels = ['hf_min', 'hf_max', 'hf_mean', 'hf_median',
                        'hf_std', 'hf_skewness', 'hf_kurtosis', 'hf_peak',
                        'hf_peak_position',
                        'hf_range', 'hf_energy', 'hf_quartile_range',
                        'hf_entropy']

    histogram_features = [hist_min, hist_max, hist_mean, hist_median,
                          hist_std, hist_skewness, hist_kurtosis, hist_peak,
                          hist_peak_position,
                          hist_range, energy, quartile_range, entropy]

    return histogram_features, histogram_labels


def create_histogram(data, bins):
    histogram, bins = np.histogram(data, bins)
    return histogram, bins


def get_min(data):
    # return np.amin(data)
    return np.percentile(data, 2)


def get_max(data):
    # return np.amax(data)
    return np.percentile(data, 98)


def get_median(data):
    return np.median(data)


def get_mean(data):
    return np.mean(data)


def get_std(data):
    return np.std(data)


def get_skewness(data):
    return scipy.stats.skew(data)


def get_kurtosis(data):
    return scipy.stats.kurtosis(data)


def get_peak(histogram, bins):
    return bins[np.argmax(histogram)]


def get_peak_position(histogram):
    return np.argmax(histogram)


def get_diff_in_out(image, contour):
    _, voi_voxels = cf.get_voi_voxels(contour, image)
    _, not_voi_voxels = cf.get_not_voi_voxels(contour, image)
    return np.mean(voi_voxels) - np.mean(not_voi_voxels)


def get_range(data):
    return np.percentile(data, 98) - np.percentile(data, 2)


def get_energy(data):
    energy = np.sum(np.square(data + np.min(data)))
    return energy


def get_quartile_range(data):
    return np.percentile(data, 75) - np.percentile(data, 25)


def get_entropy(hist):
    epsilon = np.spacing(1)

    sumhist = hist.sum()
    if sumhist == 0:
        return 0

    if np.min(hist) < 0:
        move_to_zero = - np.min(hist)
    else:
        move_to_zero = 0

    hist = hist + move_to_zero + epsilon  # Ensure logarithm works
    sumhist = hist.sum()
    hist = hist / float(sumhist)  # normalize
    entropy = -1.0 * np.sum(hist * np.log2(hist))
    return entropy
