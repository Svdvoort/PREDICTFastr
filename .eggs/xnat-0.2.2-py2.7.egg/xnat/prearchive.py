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
import datetime

import isodate

from .core import XNATObject
from .datatypes import to_date, to_time


class PrearchiveSession(XNATObject):
    @property
    def id(self):
        return '{}/{}/{}'.format(self.data['project'], self.data['timestamp'], self.data['name'])

    @property
    def fulldata(self):
        return self.xnat_session.get_json(self.uri)['ResultSet']['Result'][0]

    @property
    def data(self):
        return self.fulldata

    @property
    def autoarchive(self):
        return self.data['autoarchive']

    @property
    def folder_name(self):
        return self.data['folderName']

    @property
    def lastmod(self):
        lastmod_string = self.data['lastmod']
        return datetime.datetime.strptime(lastmod_string, '%Y-%m-%d %H:%M:%S.%f')

    @property
    def name(self):
        return self.data['name']

    @property
    def label(self):
        return self.name

    @property
    def prevent_anon(self):
        return self.data['prevent_anon']

    @property
    def prevent_auto_commit(self):
        return self.data['prevent_auto_commit']

    @property
    def project(self):
        return self.data['project']

    @property
    def scan_date(self):
        try:
            return to_date(self.data['scan_date'])
        except isodate.ISO8601Error:
            return None

    @property
    def scan_time(self):
        try:
            return to_time(self.data['scan_time'])
        except isodate.ISO8601Error:
            return None

    @property
    def status(self):
        return self.data['status']

    @property
    def subject(self):
        return self.data['subject']

    @property
    def tag(self):
        return self.data['tag']

    @property
    def timestamp(self):
        return self.data['timestamp']

    @property
    def uploaded(self):
        uploaded_string = self.data['uploaded']
        try:
            return datetime.datetime.strptime(uploaded_string, '%Y-%m-%d %H:%M:%S.%f')
        except ValueError:
            return None

    @property
    def scans(self):
        data = self.xnat_session.get_json(self.uri + '/scans')
        # We need to prepend /data to our url (seems to be a bug?)

        return [PrearchiveScan('{}/scans/{}'.format(self.uri, x['ID']),
                               self.xnat_session,
                               datafields=x) for x in data['ResultSet']['Result']]

    def download(self, path):
        self.xnat_session.download_zip(self.uri, path)
        return path

    def archive(self, overwrite=None, quarantine=None, trigger_pipelines=None, project=None, subject=None, experiment=None):
        query = {'src': self.uri}

        if overwrite is not None:
            if overwrite not in ['none', 'append', 'delete']:
                raise ValueError('Overwrite should be none, append or delete!')
            query['overwrite'] = overwrite

        if quarantine is not None:
            if isinstance(quarantine, bool):
                if quarantine:
                    query['quarantine'] = 'true'
                else:
                    query['quarantine'] = 'false'
            else:
                raise TypeError('Quarantine should be a boolean')

        if trigger_pipelines is not None:
            if isinstance(trigger_pipelines, bool):
                if trigger_pipelines:
                    query['triggerPipelines'] = 'true'
                else:
                    query['triggerPipelines'] = 'false'
            else:
                raise TypeError('trigger_pipelines should be a boolean')

        # Change the destination of the session
        # BEWARE the dest argument is completely ignored, but there is a work around:
        # HACK: See https://groups.google.com/forum/#!searchin/xnat_discussion/prearchive$20archive$20service/xnat_discussion/hwx3NOdfzCk/rQ6r2lRpZjwJ
        if project is not None:
            query['project'] = project

        if subject is not None:
            query['subject'] = subject

        if experiment is not None:
            query['session'] = experiment

        response = self.xnat_session.post('/data/services/archive', query=query)
        object_uri = response.text.strip()

        self.clearcache()  # Make object unavailable
        return self.xnat_session.create_object(object_uri)

    def delete(self, async=None):
        query = {'src': self.uri}

        if async is not None:
            if isinstance(async, bool):
                if async:
                    query['async'] = 'true'
                else:
                    query['async'] = 'false'
            else:
                raise TypeError('async should be a boolean')

        response = self.xnat_session.post('/data/services/prearchive/delete', query=query)
        self.clearcache()
        return response

    def rebuild(self, async=None):
        query = {'src': self.uri}

        if async is not None:
            if isinstance(async, bool):
                if async:
                    query['async'] = 'true'
                else:
                    query['async'] = 'false'
            else:
                raise TypeError('async should be a boolean')

        response = self.xnat_session.post('/data/services/prearchive/rebuild', query=query)
        self.clearcache()
        return response

    def move(self, new_project, async=None):
        query = {'src': self.uri,
                 'newProject': new_project}

        if async is not None:
            if isinstance(async, bool):
                if async:
                    query['async'] = 'true'
                else:
                    query['async'] = 'false'
            else:
                raise TypeError('async should be a boolean')

        response = self.xnat_session.post('/data/services/prearchive/move', query=query)
        self.clearcache()
        return response


class PrearchiveScan(XNATObject):
    def __init__(self, uri, xnat_session, id_=None, datafields=None, parent=None, fieldname=None):
        super(PrearchiveScan, self).__init__(uri=uri,
                                             xnat_session=xnat_session,
                                             id_=id_,
                                             datafields=datafields,
                                             parent=parent,
                                             fieldname=fieldname)

        self._fulldata = {'data_fields': datafields}

    @property
    def series_description(self):
        return self.data['series_description']

    def download(self, path):
        self.xnat_session.download_zip(self.uri, path)
        return path

    @property
    def fulldata(self):
        return self._fulldata


class Prearchive(object):
    def __init__(self, xnat_session):
        self._xnat_session = xnat_session

    @property
    def xnat_session(self):
        return self._xnat_session

    def sessions(self, project=None):
        if project is None:
            uri = '/data/prearchive/projects'
        else:
            uri = '/data/prearchive/projects/{}'.format(project)

        data = self.xnat_session.get_json(uri)
        # We need to prepend /data to our url (seems to be a bug?)
        return [PrearchiveSession('/data{}'.format(x['url']), self.xnat_session) for x in data['ResultSet']['Result']]
