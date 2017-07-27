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

from __future__ import absolute_import
from __future__ import unicode_literals
import fnmatch


class Inspect(object):
    def __init__(self, xnat_session):
        self._xnat_session = xnat_session

    @property
    def xnat_session(self):
        return self._xnat_session

    def datatypes(self, pattern='*', fields_pattern=None):
        elements = self.xnat_session.get_json('/data/search/elements')

        elements = [x['ELEMENT_NAME'] for x in elements['ResultSet']['Result']]

        # Filter fields using pattern
        if '*' in pattern or '?' in pattern:
            elements = [field for field in elements if fnmatch.fnmatch(field, pattern)]

        if fields_pattern is None:
            return elements
        else:
            return [field for element in elements for field in self.datafields(datatype=element, pattern=fields_pattern)]

    def datafields(self, datatype, pattern='*', prepend_type=True):
        search_fields = self.xnat_session.get_json('/data/search/elements/{}'.format(datatype))

        # Select data from JSON
        search_fields = [x['FIELD_ID'] for x in search_fields['ResultSet']['Result']]

        # Filter fields using pattern
        if '*' in pattern or '?' in pattern:
            search_fields = [field for field in search_fields if fnmatch.fnmatch(field, pattern)]

        # Filter fields for unwanted fields
        search_fields = [field for field in search_fields if '=' not in field and 'SHARINGSHAREPROJECT' not in field]

        return ['{}/{}'.format(datatype, field) if prepend_type else field for field in search_fields]

