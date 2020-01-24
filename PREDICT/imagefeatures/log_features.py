#!/usr/bin/env python

# Copyright 2017-2020 Biomedical Imaging Group Rotterdam, Departments of
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
# import imagefeatures.histogram_features as hf
# import PREIDCT.helpers.sitk_helper as sh
import SimpleITK as sitk
import numpy as np

N_BINS = 50


def get_log_features(image, mask, parameters=dict()):
    '''
    Compute features by filtering an image with a Laplacian of Gaussian (LoG)
    filter, after which histogram features are extracted.

    Parameters
    ----------
    image: numpy array, mandatory
            Image array from which the features are extracted.

    mask: numpy array, mandatory
            ROI to be used for feature extraction.

    parameters: dictionary, optional
            Contains the parameters for feature computation. Currently can only
            include a list of sigma values to be used for the LoG filter.
            Default values for sigma are [1, 5, 10].

    Returns
    ----------
    LoG_features: list
            Contains the values for all extracted features.

    LoG_labels: list
            Contains the labels for all extracted features. Each entry
            corresponds to the element with the same index from the
            LoG_features object.

    '''
    # Convert image to array and get size
    image = sitk.GetImageFromArray(image)
    im_size = image.GetSize()

    # mask = sh.GetImageFromArray(mask.astype(np.uint8))
    if "sigma" in parameters.keys():
        sigma = parameters["sigma"]
    else:
        sigma = [1, 5, 10]

    # Make a dummy
    LoG_features = list()
    LoG_labels = list()

    # Create LoG filter object
    LoGFilter = sitk.LaplacianRecursiveGaussianImageFilter()
    LoGFilter.SetNormalizeAcrossScale(True)

    # Iterate over sigmas
    for i_index, i_sigma in enumerate(sigma):
        LoGFilter.SetSigma(i_sigma)

        if len(im_size) == 2:
            # 2D Image
            LoG_image = LoGFilter.Execute(image)
            LoG_image = sitk.GetArrayFromImage(LoG_image)
        else:
            # 3D Image
            LoG_image = np.zeros([image.GetSize()[0], image.GetSize()[2], image.GetSize()[1]])

            # LoG Feature needs a minimum of 4 voxels in each direction
            if not any(t < 4 for t in im_size):
                # Iterate over slices
                for i_slice in range(0, image.GetSize()[0]):
                    # Compute LoG Filter image
                    LoG_image_temp = LoGFilter.Execute(image[i_slice, :, :])
                    LoG_image[i_slice, :, :] = sitk.GetArrayFromImage(LoG_image_temp)

        # Get histogram features of LoG image for full tumor
        masked_voxels = ih.get_masked_voxels(LoG_image, mask)
        histogram_features, histogram_labels = hf.get_histogram_features(masked_voxels, N_BINS)
        histogram_labels = [l.replace('hf_', 'logf_') for l in histogram_labels]
        LoG_features.extend(histogram_features)
        final_feature_names = [feature_name + '_sigma' + str(i_sigma) for feature_name in histogram_labels]
        LoG_labels.extend(final_feature_names)

    return LoG_features, LoG_labels
