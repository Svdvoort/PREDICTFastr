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
import pandas as pd

import PREDICT.helpers.contour_functions as cf


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


def get_peak_position(histogram, bins):
    return np.amax(histogram)


def get_diff_in_out(image, contour):
    _, voi_voxels = cf.get_voi_voxels(contour, image)
    _, not_voi_voxels = cf.get_not_voi_voxels(contour, image)
    return np.mean(voi_voxels) - np.mean(not_voi_voxels)


def get_range(data):
    return np.percentile(data, 98) - np.percentile(data, 2)


def get_histogram_features(data, N_bins):
    temp_histogram, temp_bins = create_histogram(data, N_bins)
    hist_min = get_min(data)
    hist_max = get_max(data)
    hist_mean = get_mean(data)
    hist_median = get_median(data)
    hist_std = get_std(data)
    hist_skewness = get_skewness(data)
    hist_kurtosis = get_kurtosis(data)
    hist_range = get_range(data)
    hist_peak = get_peak_position(temp_histogram, temp_bins)

    panda_labels = ['hist_min', 'hist_max', 'hist_mean', 'hist_median',
                    'hist_std', 'hist_skewness', 'hist_kurtosis', 'hist_peak',
                    'hist_range']

    histogram_features = [hist_min, hist_max, hist_mean, hist_median,
                          hist_std, hist_skewness, hist_kurtosis, hist_peak,
                          hist_range]

    panda_dict = dict(zip(panda_labels, histogram_features))

    hist_dict = dict()
    hist_dict['all'] = pd.Series(panda_dict)

    histogram_features = pd.Series(hist_dict)

    return histogram_features
