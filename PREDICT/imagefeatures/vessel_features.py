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
import PREDICT.imagefeatures.histogram_features as hf
import PREDICT.helpers.image_helper as ih
from skimage.filters import frangi
import numpy as np
from skimage import morphology

N_BINS = 50


def get_vessel_features(image, mask, parameters=dict()):
    # Alternatively, one could use the pxehancement function
    if "scale_range" in parameters.keys():
        scale_range = parameters["scale_range"]
    else:
        scale_range = [1, 10]

    # Convert to tuple format accepted by the function
    scale_range = [(scale_range[0], scale_range[1])]

    if "scale_step" in parameters.keys():
        scale_step = parameters["scale_step"]
    else:
        scale_step = [2]

    if "radius" in parameters.keys():
        radius = parameters["radius"]
    else:
        radius = 5

    # Make a dummy
    Frangi_features = list()
    Frangi_labels = list()

    # Create different masks for edge and inner tumor
    disk = morphology.disk(radius)
    mask_edge = np.zeros(mask.shape)
    mask_inner = np.zeros(mask.shape)
    for ind in range(mask.shape[2]):
        mask_e = morphology.binary_erosion(mask[:, :, ind], disk)
        mask_edge[:, :, ind] = np.logical_or(mask[:, :, ind], mask_e)
        mask_inner[:, :, ind] = mask_e

    for i_index, (i_sr, i_ss) in enumerate(zip(scale_range, scale_step)):
        # Compute Frangi Filter image
        Frangi_image = np.zeros(image.shape)
        for i_slice in range(0, image.shape[2]):
            # Note: conversion to uint8, as skimage cannot filter certain float images
            Frangi_image[:, :, i_slice] = frangi(image[:, :, i_slice].astype(np.uint8), scale_range=i_sr, scale_step=i_ss)

        # Get histogram features of Frangi image for full tumor
        masked_voxels = ih.get_masked_voxels(Frangi_image, mask)
        if masked_voxels.size == 0:
            print("[PREDICT Warning] Vessel features, fully empty. Using zeros.")
            masked_voxels = [0]
        histogram_features, histogram_labels = hf.get_histogram_features(masked_voxels, N_BINS)
        histogram_labels = [l.replace('hf_', 'vf_Frangi_full_') for l in histogram_labels]
        Frangi_features.extend(histogram_features)
        final_feature_names = [feature_name + '_SR' + str(i_sr) + '_SS' + str(i_ss) for feature_name in histogram_labels]
        Frangi_labels.extend(final_feature_names)

        # Get histogram features of Frangi image for edge
        masked_voxels = ih.get_masked_voxels(Frangi_image, mask_edge)
        if masked_voxels.size == 0:
            print("[PREDICT Warning] Vessel features, edge area empty. Using zeros.")
            masked_voxels = [0]
        histogram_features, histogram_labels = hf.get_histogram_features(masked_voxels, N_BINS)
        histogram_labels = [l.replace('hf_', 'vf_Frangi_edge_') for l in histogram_labels]
        Frangi_features.extend(histogram_features)
        final_feature_names = [feature_name + '_SR' + str(i_sr) + '_SS' + str(i_ss) for feature_name in histogram_labels]
        Frangi_labels.extend(final_feature_names)

        # Get histogram features of Frangi image inside tumor only
        masked_voxels = ih.get_masked_voxels(Frangi_image, mask_inner)
        if masked_voxels.size == 0:
            print("[PREDICT Warning] Vessel features, inner area empty. Using zeros.")
            masked_voxels = [0]
        histogram_features, histogram_labels = hf.get_histogram_features(masked_voxels, N_BINS)
        histogram_labels = [l.replace('hf_', 'vf_Frangi_inner_') for l in histogram_labels]
        Frangi_features.extend(histogram_features)
        final_feature_names = [feature_name + '_SR' + str(i_sr) + '_SS' + str(i_ss) for feature_name in histogram_labels]
        Frangi_labels.extend(final_feature_names)

    return Frangi_features, Frangi_labels
