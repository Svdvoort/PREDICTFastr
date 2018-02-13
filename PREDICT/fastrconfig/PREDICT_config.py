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

import site
import os
import sys


# Get directory in which packages are installed
try:
    packagedir = site.getsitepackages()[0]
except AttributeError:
    # Inside virtualenvironment, so getsitepackages doesnt work.
    paths = sys.path
    for p in paths:
        if os.path.isdir(p) and os.path.basename(p) == 'site-packages':
            packagedir = p

# packagedir is the path in which PREDICT is installed. Default on Linux is /usr/local/lib/python2.7/site-packages
tools_path = [os.path.join(packagedir, 'PREDICT', 'fastr_tools')] + tools_path
