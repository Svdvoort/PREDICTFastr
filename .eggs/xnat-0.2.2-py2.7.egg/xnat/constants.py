# Copyright 2011-2015 Biomedical Imaging Group Rotterdam, Departments of
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

TYPE_HINTS = {
    'demographics': 'xnat:demographicData',
    'investigator': 'xnat:investigatorData',
    'metadata': 'xnat:subjectMetadata',
    'pi': 'xnat:investigatorData',
    'studyprotocol': 'xnat:studyProtocol',
    'validation': 'xnat:validationData',
    'baseimage': 'xnat:abstractResource',
    'projects': 'xnat:projectData',
    'subjects': 'xnat:subjectData',
    'experiments': None,  # Can be many types, need to check each time
    'scans': None,  # Can be many types, need to check each time
    'resources': None,  # Can be many types, need to check each time
    'assessors': None,   # Can be many types, need to check each time
    'reconstructions': None,  # Can be many types, need to check each time
    'files': 'xnat:fileData',
}

FIELD_HINTS = {
    'xnat:projectData': 'projects',
    'xnat:subjectData': 'subjects',
    'xnat:experimentData': 'experiments',
    'xnat:imageScanData': 'scans',
    'xnat:reconstructedImageData': 'reconstructions',
    'xnat:imageAssessorData': 'assessors',
    'xnat:abstractResource': 'resources',
    'xnat:fileData': 'files',
}

# The following xsi_types have a listing of objects with own REST paths the
# other are nested in the xml of their parent.
OBJECT_LISTINGS = {
    'xnat:projectData',
    'xnat:subjectData',
    'xnat:subjectAssessorData',
    'xnat:reconstructedImageData',
    'xnat:imageAssessorData',
    'xnat:imageScanData',
    'xnat:abstractResource',
    'xnat:fileData',
}

SECONDARY_LOOKUP_FIELDS = {
    'xnat:projectData': 'name',
    'xnat:subjectData': 'label',
    'xnat:experimentData': 'label',
    'xnat:imageScanData': 'type',
    'xnat:abstractResource': 'label',
    'xnat:fileData': 'name'
}
