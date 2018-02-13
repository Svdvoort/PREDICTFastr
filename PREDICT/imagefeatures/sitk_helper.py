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
import SimpleITK as sitk


def GetImageFromArray(array):
    """
    GetImageFromArray converts a numpy array to an ITK image, while ensuring
    the orientation is kept the same.

    Args:
        array (numpy array): 2D or 3D array of image

    Returns:
        ITK image

    """

    if len(array.shape) == 3:
        array = np.transpose(array, [2, 1, 0])
    elif len(array.shape) == 2:
        array = np.transpose(array, [1, 0])
    return sitk.GetImageFromArray(array)


def GetArrayFromImage(image):
    """
    GetArrayFromImage converts an ITK image to a numpy array, while ensuring
    the orientation is kept the same.

    Args:
        image (ITK image): 2D or 3D ITK image

    Returns:
        numpy array

    """

    array = sitk.GetArrayFromImage(image)
    if len(array.shape) == 3:
        array = np.transpose(array, [2, 1, 0])
    elif len(array.shape) == 2:
        array = np.transpose(array, [1, 0])
    return array
