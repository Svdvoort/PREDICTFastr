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
import PREDICT.helpers.sitk_helper as sitkh


def get_masked_slices_image(image_array, mask_array):
    '''
    For a 3-D array, get only those axial slices which contains non-zero numbers
    in the mask.

    Parameters
    ---------
    image_array: numpy array, mandatory
            Array containing the array to be masked.

    mask_array: numpy array, mandatory
            Array containing the mask.

    Returns
    ---------
    image_array: numpy array, mandatory
            Array containing the masked image array.

    mask_array: numpy array, mandatory
            Array containing the masked mask array.

    '''
    mask_array = mask_array.astype(np.bool)

    mask_slices = np.any(mask_array, axis=(0, 1))
    try:
        image_array = image_array[:, :, mask_slices]
        mask_array = mask_array[:, :, mask_slices]
    except IndexError:
        print("Note: Mask indexing does not match image!")
        mask_slices = mask_slices[0:image_array.shape[2]]
        image_array = image_array[:, :, mask_slices]
        mask_array = mask_array[:, :, mask_slices]

    return image_array, mask_array


def get_masked_voxels(image_array, mask_array):
    '''
    Returns only those voxels of an array for which the mask array value is True.

    Parameters
    ---------
    image_array: numpy array, mandatory
            Array containing the array to be masked.

    mask_array: numpy array, mandatory
            Array containing the mask.

    Returns
    ----------
    masked_voxels: numpy array
            1D Array containing only those voxels for which the mask was True.

    '''
    mask_array = mask_array.astype(np.bool)

    mask_array = mask_array.flatten()
    image_array = image_array.flatten()

    masked_voxels = image_array[mask_array]

    return masked_voxels


def get_masked_slices_mask(mask_image):
    '''
    Remove the axial slices in a 3-D mask array for which there are non-zero
    elements.

    Parameters
    ----------
    mask_image: numpy array, mandatory
            Boolean 3D array.

    Returns
    -----------
    mask_sliced: numpy array, mandatory
            Array containing only those slices of which there are non-zero
            elements in the mask.

    '''
    mask_array = sitkh.GetArrayFromImage(mask_image)

    # Filter out slices where there is no mask (need actual index here)
    mask_slices = np.flatnonzero(np.any(mask_array, axis=(0, 1)))

    if len(mask_slices) == 1:
        mask_sliced = mask_image[:, :, mask_slices[0]:(mask_slices[0] + 1)]
    else:
        mask_sliced = mask_image[:, :, mask_slices[0]:mask_slices[-1] + 1]

    return mask_sliced
