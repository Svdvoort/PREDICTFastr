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


import imagefeatures.get_features as gf
import IOparser.config_io_CalcFeatures as config_io
import IOparser.file_io as IO
import pandas as pd
import SimpleITK as sitk
import numpy as np
from skimage import morphology
import scipy.ndimage as nd
import os
import dicom as pydicom

# There is a small difference between the contour and image origin and spacing
# Fix this by setting a slightly larger, but still reasonable tolerance
# (Defaults to around 8e-7, which seems very small)
sitk.ProcessObject.SetGlobalDefaultCoordinateTolerance(5e-5)


def CalcFeatures(image, segmentation, parameters, output,
                 metadata_file=None, semantics_file=None, verbose=True):

    # Load variables from the confilg file
    config = config_io.load_config(parameters)

    # Calculate the image features
    gabor_settings = config['ImageFeatures']['gabor_settings']
    image_type = config['ImageFeatures']['image_type']

    panda_labels = ['image_type', 'gabor_settings', 'feature_values',
                    'feature_labels']

    print('Calculating image features!')
    image_data = load_images(image, image_type, metadata_file, semantics_file)

    # contours = selectsegmentation(segmentation, config["ImageFeatures"])
    if type(segmentation) is list:
        segmentation = ''.join(segmentation)
    contours = [sitk.ReadImage(segmentation)]

    feature_values, feature_labels =\
        gf.get_image_features(image_data, contours,
                              gabor_settings, 0, False,
                              config["ImageFeatures"], output)

    panda_data = pd.Series([image_type, gabor_settings, feature_values,
                            feature_labels],
                           index=panda_labels,
                           name='Image features'
                           )

    print('Saving image features')
    panda_data.to_hdf(output, 'image_features')

    if verbose:
        for v, k in zip(feature_values, feature_labels):
            print k, v


def selectsegmentation(segmentation, config):
        contours = list()
        if type(segmentation) is list:
            segmentation = ''.join(segmentation)

        # Convert to binary image and clean up small errors/areas
        # TODO: More robust is to do this with labeling and select largest blob
        contour = sitk.ReadImage(segmentation)
        contour = sitk.GetArrayFromImage(contour)
        contour = nd.binary_fill_holes(contour)
        contour = contour.astype(bool)
        contour = morphology.remove_small_objects(contour, min_size=2, connectivity=2, in_place=False)

        # Expand contour depending on settings
        if config['segmentation']['type'] == 'Ring':
            radius = int(config['segmentation']['radius'])
            disk = morphology.disk(radius)

            # Dilation with radius
            for ind in range(contour.shape[2]):
                contour_d = morphology.binary_dilation(contour[:, :, ind], disk)
                contour_e = morphology.binary_erosion(contour[:, :, ind], disk)
                contour[:, :, ind] = np.subtract(contour_d, contour_e)

        contour = contour.astype(np.uint8)
        contour = sitk.GetImageFromArray(contour)
        contours.append(contour)

        return contours


def load_images(image_folder, image_type, metadata_file, semantics_file):
    # TODO: DTI metadata extraction, now only simple metadata
    images = list()
    metadata = list()
    semantics = list()
    if type(image_folder) is list:
        image_folder = ''.join(image_folder)

    if type(metadata_file) is list:
        metadata_file = ''.join(metadata_file)

    if type(semantics_file) is list:
        semantics_file = ''.join(semantics_file)

    if image_folder[-6::] == 'nii.gz':
        metadata_temp = None
        image_temp = sitk.ReadImage(image_folder)

        if 'MR' in image_type:
            # Normalize image
            image_temp = sitk.Normalize(image_temp)

        if metadata_file is not None:
            metadata_temp = pydicom.read_file(metadata_file)
            metadata_temp.pixel_array = None  # save memory

    else:
        # Assume DICOM
        if 'MR' in image_type:
            image_temp, metadata_temp = IO.load_dicom(image_folder)
            # Normalize image
            image_temp = sitk.Normalize(image_temp)
        elif 'DTI' in image_type:
            image_temp, metadata_temp = IO.load_DTI(image_folder)
        elif 'CT' in image_type:
            image_temp, metadata_temp = IO.load_dicom(image_folder)
            # Convert intensity to Hounsfield units
            image_temp = image_temp*metadata_temp.RescaleSlope + metadata_temp.RescaleIntercept

    if semantics_file is not None:
        _, file_extension = os.path.splitext(semantics_file)
        if file_extension == '.txt':
            # TODO: fix that this readout converges to list types per sem
            semantics_temp = np.loadtxt(semantics_file, np.str)
        elif file_extension == '.csv':
            import csv
            semantics_temp = dict()
            with open(semantics_file, 'rb') as f:
                reader = csv.reader(f)
                for num, row in enumerate(reader):
                    if num == 0:
                        header = row
                        if header[0] != 'Patient':
                            raise AssertionError('First column should be patient ID!')

                        keys = list()
                        for key in header:
                            semantics_temp[key] = list()
                            keys.append(key)
                    else:
                        for column in range(len(row)):
                            if column > 0:
                                semantics_temp[keys[column]].append(float(row[column]))
                            else:
                                semantics_temp[keys[column]].append(row[column])
    else:
        semantics_temp = None

    images.append(image_temp)
    metadata.append(metadata_temp)
    semantics.append(semantics_temp)

    image_data = {'images': images, 'metadata': metadata, 'semantics': semantics, 'image_type': image_type}
    return image_data
