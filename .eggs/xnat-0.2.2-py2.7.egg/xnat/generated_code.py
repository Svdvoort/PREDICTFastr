
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
import os
import tempfile  # Needed by generated code
from zipfile import ZipFile  # Needed by generated code

from xnat import search
from xnat.core import XNATBaseObject, XNATSubObject, XNATListing, caching
from xnat.utils import mixedproperty


SESSION = None


def current_session():
    return SESSION


# These mixins are to set the xnat_session automatically in all created classes
class XNATObjectMixin(XNATBaseObject):
    @mixedproperty
    def xnat_session(self):
        return current_session()

    @classmethod
    def query(cls, *constraints):
        query = search.Query(cls._XSI_TYPE, cls.xnat_session)

        # Add in constraints immediatly
        if len(constraints) > 0:
            query = query.filter(*constraints)

        return query


class XNATSubObjectMixin(XNATSubObject):
    @mixedproperty
    def xnat_session(self):
        return current_session()


class FileData(XNATObjectMixin):
    SECONDARY_LOOKUP_FIELD = "name"
    _XSI_TYPE = 'xnat:fileData'

    def __init__(self, uri, xnat_session, id_=None, datafields=None, name=None, parent=None, fieldname=None):
        super(FileData, self).__init__(uri=uri,
                                   xnat_session=xnat_session,
                                   id_=id_,
                                   datafields=datafields,
                                   parent=parent,
                                   fieldname=fieldname)
        if name is not None:
            self._name = name

    @property
    def name(self):
        return self._name

    def delete(self):
        self.xnat_session.delete(self.uri)

    def download(self, path, verbose=True):
        self.xnat_session.download(self.uri, path, verbose=verbose)

    def download_stream(self, target_stream, verbose=False):
        self.xnat_session.download_stream(self.uri, target_stream, verbose=verbose)

    @property
    @caching
    def size(self):
        response = self.xnat_session.head(self.uri)
        return response.headers['Content-Length']


# Empty class lookup to place all new lookup values
XNAT_CLASS_LOOKUP = {
    "xnat:fileData": FileData,
}


# The following code represents the data structure of the XNAT server
# It is automatically generated using
# - file:///home/hachterberg/dev/xnat/xnatpy/xnat/xnat.xsd


