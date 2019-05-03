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

import pandas as pd
import SimpleITK as sitk
import numpy as np
import os
import pydicom
import PREDICT.addexceptions as ae
import PREDICT.imagefeatures.get_features as gf
import PREDICT.IOparser.config_io_CalcFeatures as config_io
import PREDICT.IOparser.file_io as IO

# There is a small difference between the contour and image origin and spacing
# Fix this by setting a slightly larger, but still reasonable tolerance
# (Defaults to around 8e-7, which seems very small)
sitk.ProcessObject.SetGlobalDefaultCoordinateTolerance(5e-5)


def CalcFeatures(image, segmentation, parameters, output,
                 metadata_file=None, semantics_file=None, verbose=True):
    '''
    Calculate features from a ROI of an image. This function serves as a wrapper
    around the get_features function from the imagefeatures folder. It
    reads all inputs, processes it through the get_features function per image
    and ROI and writes the output to HDF5 files.

    Parameters
    ----------
    image: string, mandatory
            path referring to image file. Should be a format compatible
            with ITK, e.g. .nii, .nii.gz, .mhd, .raw, .tiff, .nrrd or
            a DICOM folder.

    segmentation: string, mandatory
            path referring to segmentation file. Should be a format compatible
            with ITK, e.g. .nii, .nii.gz, .mhd, .raw, .tiff, .nrrd.

    parameters: string, mandatory,
            path referring to a .ini file containing the parameters
            used for feature extraction. See the Github Wiki for the possible
            fields and their description.

    output: string, mandatory
            path referring to the .hdf5 file to which the output should be written.

    metadata_file: string, optional
            path referring to a .dcm file from which the patient features will
            be extracted.

    semantics_file: string, optional
            path referring to a .csv file from which the semantic features will
            be extracted. See the Github Wiki for the correct format.

    verbose: boolean, default True
            print final feature values and labels to command line or not.

    '''
    # Load variables from the confilg file
    config = config_io.load_config(parameters)

    # Calculate the image features
    parameters = config['ImageFeatures']['parameters']
    image_type = config['ImageFeatures']['image_type']

    panda_labels = ['image_type', 'parameters', 'feature_values',
                    'feature_labels']

    print('Loading inputs.')
    # Read the image data, metadata and semantics
    image_data = load_images(image, image_type, metadata_file, semantics_file)

    # Read the contour
    print('Load segmentation.')
    if type(segmentation) is list:
        segmentation = ''.join(segmentation)

    contour = sitk.ReadImage(segmentation)

    # FIXME: Bug in some of our own segmentations. Shouldnt occur in normal usage
    szi = image_data['images'].GetSize()
    szs = contour.GetSize()
    if szi != szs:
        message = ('Shapes of image({}) and mask ({}) do not match!').format(str(szi), str(szs))
        print(message)
        # FIXME: Now excluding last slice, not an elegant solution
        contour = sitk.GetArrayFromImage(contour)
        contour = contour[0:-1, :, :]
        contour = sitk.GetImageFromArray(contour)

        # Check if excluding last slice fixed the problem
        szs = contour.GetSize()
        if szi != szs:
            message = ('Shapes of image({}) and mask ({}) do not match!').format(str(szi), str(szs))
            raise ae.PREDICTIndexError(message)
        else:
            print("['FIXED'] Excluded last slice.")

    # Extract the actual features
    print('Calculating image features.')
    feature_values, feature_labels =\
        gf.get_image_features(image_data, contour,
                              parameters,
                              config["ImageFeatures"],
                              config["General"],
                              output)

    # Convert to pandas Series and save as hdf5
    panda_data = pd.Series([image_type, parameters, feature_values,
                            feature_labels],
                           index=panda_labels,
                           name='Image features'
                           )

    print('Saving image features')
    panda_data.to_hdf(output, 'image_features')

    # If required, print output feature values
    if verbose:
        print('Features extracted:')
        for v, k in zip(feature_values, feature_labels):
            print(k, v)

    return panda_data


def load_images(image_file, image_type, metadata_file=None,
                semantics_file=None):
    '''
    Load ITK images, the corresponding DICOM file for the metadata, a file
    containing the semantics and converts them to Python objects.

    Parameters
    ----------
    image_file: string, mandatory
            path referring to image file. Should be a format compatible
            with ITK, e.g. .nii, .nii.gz, .mhd, .raw, .tiff, .nrrd. or a
            DICOM folder.

    image_type: string, mandatory
            defines the modality of the scan used. Different loading functions
            are used for different modalities.

    metadata_file: string, optional
            path referring to a DICOM file. Used to extract metadata features.

    semantics_file: string, optional
            path referring to a CSV file. Used to extract semantic features.

    '''
    # Convert the input arguments to strings if given as lists
    if type(image_file) is list:
        image_file = ''.join(image_file)

    if type(metadata_file) is list:
        metadata_file = ''.join(metadata_file)

    if type(semantics_file) is list:
        semantics_file = ''.join(semantics_file)

    # Read the input image based on the filetype provided
    print('Load image and metadata file.')
    extension = os.path.splitext(image_file)
    if extension == '.dcm':
        # Single DICOM, so convert back to a list to use load_dicom
        image_file = [image_file]
        if 'MR' in image_type:
            image, metadata = IO.load_dicom(image_file)
        elif 'DTI' in image_type:
            image, metadata = IO.load_DTI(image_file)
        elif 'CT' in image_type:
            image, metadata = IO.load_dicom(image_file)

            # Convert intensity to Hounsfield units
            image = image*metadata.RescaleSlope +\
                metadata.RescaleIntercept

    elif not os.path.isfile(image_file):
        # Assume input is a DICOM folder
        if 'MR' in image_type:
            image, metadata = IO.load_dicom(image_file)
        elif 'DTI' in image_type:
            image, metadata = IO.load_DTI(image_file)
        elif 'CT' in image_type:
            image, metadata = IO.load_dicom(image_file)

    else:
        # Since input is only an image file, temporary set metadata to None
        metadata = None
        image = sitk.ReadImage(image_file)

        if metadata_file is not None:
            metadata = pydicom.read_file(metadata_file)
            metadata.PixelArray = None  # save memory

    # Read the semantics CSV and match values to the image file
    print('Load semantics file.')
    if semantics_file is not None:
        _, file_extension = os.path.splitext(semantics_file)
        if file_extension == '.txt':
            # TODO: fix that this readout converges to list types per sem
            semantics = np.loadtxt(semantics_file, np.str)
        elif file_extension == '.csv':
            data = pd.read_csv(semantics_file, header=0)
            header = data.keys()
            if header[0] != 'Patient':
                raise ae.PREDICTAssertionError('First column of the semantics file should be patient ID!')

            semantics = {k: data[k].values.tolist() for k in data.keys()}
    else:
        semantics = None

    image_data = {'images': image, 'metadata': metadata, 'semantics': semantics, 'image_type': image_type}
    return image_data
