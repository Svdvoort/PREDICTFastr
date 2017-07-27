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
import os


def load_config():
    settings = configparser.ConfigParser()
    settings.read(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'config.ini'))

    settings_dict = {'Joblib': dict()}

    # TODO: Generate a PREDICT config file in ~/.predict upon installation and parse
    # settings_dict['Joblib']['njobs'] =\
    #     int(settings['Joblib']['njobs'])
    #
    # settings_dict['Joblib']['backend'] =\
    #     str(settings['Joblib']['backend'])
    settings_dict['Joblib']['njobs'] = 6

    settings_dict['Joblib']['backend'] = "threading"

    return settings_dict
