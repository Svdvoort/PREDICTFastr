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
import mimetypes

from .prearchive import PrearchiveSession


class Services(object):
    def __init__(self, xnat_session):
        self._xnat_session = xnat_session

    @property
    def xnat_session(self):
        return self._xnat_session

    def import_(self, path, overwrite=None, quarantine=False, destination=None, project=None, subject=None, experiment=None, content_type=None):
        query = {}
        if overwrite is not None:
            if overwrite not in ['none', 'append', 'delete']:
                raise ValueError('Overwrite should be none, append or delete!')
            query['overwrite'] = overwrite

        if quarantine:
            query['quarantine'] = 'true'

        if destination is not None:
            query['dest'] = destination

        if project is not None:
            query['project'] = project

        if subject is not None:
            query['subject'] = subject

        if experiment is not None:
            query['session'] = experiment

        # Get mimetype of file
        if content_type is None:
            content_type, transfer_encoding = mimetypes.guess_type(path)

        uri = '/data/services/import'
        response = self.xnat_session.upload(uri=uri, file_=path, query=query, content_type=content_type, method='post')

        if response.status_code != 200:
            raise XNATResponseError('The response for uploading was ({}) {}'.format(response.status_code, response.text))

        # Create object, the return text should be the url, but it will have a \r\n at the end that needs to be stripped
        response_text = response.text.strip()
        if response_text.startswith('/data/prearchive'):
            return PrearchiveSession(response_text, self.xnat_session)
        else:
            return self.xnat_session.create_object(response_text)
