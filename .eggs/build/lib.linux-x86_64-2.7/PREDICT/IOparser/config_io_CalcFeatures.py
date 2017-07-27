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

import configparser
import numpy as np


def load_config(config_file_path):
    settings = configparser.ConfigParser()
    settings.read(config_file_path)

    settings_dict = {'ImageFeatures': dict(), 'DataPaths': dict()}

    settings_dict['ImageFeatures']['image_type'] =\
        [str(item).strip() for item in
         settings['ImageFeatures']['image_type'].split(',')]

    settings_dict['ImageFeatures']['texture'] =\
        str(settings['ImageFeatures']['texture'])

    settings_dict['ImageFeatures']['coliage'] =\
        str(settings['ImageFeatures']['coliage'])

    # settings_dict['ImageFeatures']['patient'] =\
    #     str(settings['ImageFeatures']['patient'])

    settings_dict['ImageFeatures']['orientation'] =\
        settings['ImageFeatures'].getboolean('orientation')

    settings_dict['ImageFeatures']['slicer'] =\
        settings['ImageFeatures'].getboolean('slicer')

    # Segmentation settings
    settings_dict['ImageFeatures']['segmentation'] = dict()
    settings_dict['ImageFeatures']['segmentation']['type'] =\
        str(settings['ImageFeatures']['segtype'])

    settings_dict['ImageFeatures']['segmentation']['radius'] =\
        int(settings['ImageFeatures']['segradius'])

    # Gabor settings
    settings_dict['ImageFeatures']['gabor_settings'] = dict()

    gabor_frequencies = [str(item).strip() for item in
                         settings['ImageFeatures']['gabor_frequencies']
                         .split(',')]
    gabor_frequencies = np.asarray(gabor_frequencies).astype(np.float)

    gabor_angles = [str(item).strip() for item in
                    settings['ImageFeatures']['gabor_angles']
                    .split(',')]
    gabor_angles = np.asarray(gabor_angles).astype(np.float)
    # Convert gabor angle to radians from angles
    gabor_angles = np.radians(gabor_angles)

    settings_dict['ImageFeatures']['gabor_settings']['gabor_frequencies'] =\
        gabor_frequencies

    settings_dict['ImageFeatures']['gabor_settings']['gabor_angles'] =\
        gabor_angles


    return settings_dict
