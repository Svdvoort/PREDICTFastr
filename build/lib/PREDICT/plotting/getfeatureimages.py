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


import PREDICT.IOparser.config_io_CalcFeatures as config_io
import SimpleITK as sitk
import numpy as np
import os
from PREDICT.CalcFeatures import load_images
import PREDICT.helpers.image_helper as ih
import skimage.filters
from joblib import Parallel, delayed
import itertools
from skimage.feature import local_binary_pattern
import scipy.stats
import PREDICT.helpers.sitk_helper as sitkh
import scipy


# There is a small difference between the contour and image origin and spacing
# Fix this by setting a slightly larger, but still reasonable tolerance
# (Defaults to around 8e-7, which seems very small)
sitk.ProcessObject.SetGlobalDefaultCoordinateTolerance(5e-5)


def getfeatureimages(image, segmentation, gabor_settings=None, image_type=None,
                     parameters=None, types=['LBP'], slicenum=None, save=False):

    if parameters is not None:
        # Load variables from the confilg file
        config = config_io.load_config(parameters)

        # Calculate the image features
        gabor_settings = config['ImageFeatures']['gabor_settings']
        image_type = config['ImageFeatures']['image_type']

    print('Calculating image features!')
    image_data = load_images(image, image_type, None, None)

    if type(segmentation) is list:
        segmentation = ''.join(segmentation)
    contours = [sitk.ReadImage(segmentation)]

    # FIXME: Bug in some of our own segmentations
    szi = image_data['images'][0].GetSize()
    szs = contours[0].GetSize()
    if szi != szs:
        message = ('Shapes of image({}) and mask ({}) do not match!').format(str(szi), str(szs))
        print(message)
        # FIXME: Now excluding last slice
        c = contours[0]
        c = sitk.GetArrayFromImage(c)
        c = c[0:-1, :, :]
        contours = [sitk.GetImageFromArray(c)]

        szs = contours[0].GetSize()
        if szi != szs:
            message = ('Shapes of image({}) and mask ({}) do not match!').format(str(szi), str(szs))
            raise IndexError(message)
        else:
            print("['FIXED'] Excluded last slice.")

    # Convert to arrays and get only masked slices
    i_image = image_data['images'][0]
    i_mask = contours[0]
    i_image_array = sitkh.GetArrayFromImage(i_image)
    i_mask_array = sitkh.GetArrayFromImage(i_mask)
    i_image_array, i_mask_array = ih.get_masked_slices_image(
                i_image_array, i_mask_array)

    if slicenum is None:
        slicenum = int(i_image_array.shape[2]/2)

    i_image_array = np.squeeze(i_image_array[:, :, slicenum])
    i_mask_array = np.squeeze(i_mask_array[:, :, slicenum])

    if save:
        filename, file_extension = os.path.splitext('/path/to/somefile.ext')
    else:
        filename = None

    im = list()
    if 'LBP' in types:
        LBP_im = save_LBP_features(i_image_array, i_mask_array, filename)
        im.append(LBP_im)
    if 'Gabor' in types:
        Gabor_im = save_gabor_features(i_image_array, i_mask_array, gabor_settings, filename)
        im.append(Gabor_im)
    if 'Shape' in types:
        im.append(i_mask_array)
    if 'Histogram' in types:
        im.append(i_image_array)

    return im


def save_gabor_features(image, mask, gabor_settings, output, n_jobs=None,
                        backend=None):
    """
    Apply gabor filters to image, done in parallel.
    Note: on a cluster, where parallelisation of the gabor filters
    is not possible, use backend="threading"
    """

    if n_jobs is None:
        n_jobs = 1
    if backend is None:
        backend = 'threading'

    # Create kernel from frequencies and angles
    kernels = list(itertools.product(gabor_settings['gabor_frequencies'],
                                     gabor_settings['gabor_angles']))

    filtered = Parallel(n_jobs=n_jobs, backend=backend)(delayed(gabor_filter)
                                                        (image=image,
                                                        mask=mask,
                                                        kernel=kernel)
                                                        for kernel in
                                                        kernels)

    if output is not None:
        for i_kernel, i_image in zip(kernels, filtered):
            # Round two to decimals to reduce name
            i_kernel = [i_kernel[0], round(i_kernel[1], 2)]
            savename = output + ('_Gabor_F{}_A{}.png').format(str(i_kernel[0]),
                                                              str(i_kernel[1]))
            scipy.misc.imsave(savename, i_image)

    return filtered


def gabor_filter(image, mask, kernel):
    filtered_image, _ = skimage.filters.gabor(image,
                                              frequency=kernel[0],
                                              theta=kernel[1])
    filtered_image[~mask] = 0
    return filtered_image


def save_LBP_features(image, mask, output):
    # TODO: Should be moved to WORC
    radius = [3]
    N_points = [24]
    method = 'uniform'

    for i_index, (i_radius, i_N_points) in enumerate(zip(radius, N_points)):
        LBP_image = local_binary_pattern(image, P=i_N_points, R=i_radius, method=method)
        LBP_tumor = LBP_image
        LBP_tumor[~mask] = 0

        if output is not None:
            savename = output + ('_LBP_R{}_N{}.png').format(str(i_radius), str(i_N_points))
            scipy.misc.imsave(savename, LBP_tumor)

    return LBP_tumor
