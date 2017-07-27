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
import netrc
import os
import sys
import threading

import requests
from six.moves.urllib import parse

from . import exceptions
from .core import XNATListing, caching
from .inspect import Inspect
from .prearchive import Prearchive
from .services import Services


class XNATSession(object):
    """
    The main XNATSession session class. It keeps a connection to XNATSession alive and
    manages the main communication to XNATSession. To keep the connection alive
    there is a background thread that sends a heart-beat to avoid a time-out.

    The main starting points for working with the XNATSession server are:

    * :py:meth:`XNATSession.projects <xnat.XNATSession.projects>`
    * :py:meth:`XNATSession.subjects <xnat.XNATSession.subjects>`
    * :py:meth:`XNATSession.experiments <xnat.XNATSession.experiments>`
    * :py:meth:`XNATSession.prearchive <xnat.XNATSession.prearchive>`
    * :py:meth:`XNATSession.services <xnat.XNATSession.services>`

    .. note:: Some methods create listing that are using the :py:class:`xnat.XNATListing`
              class. They allow for indexing with both XNATSession ID and a secondary key (often the
              label). Also they support basic filtering and tabulation.

    There are also methods for more low level communication. The main methods
    are :py:meth:`XNATSession.get <xnat.XNATSession.get>`, :py:meth:`XNATSession.post <xnat.XNATSession.post>`,
    :py:meth:`XNATSession.put <xnat.XNATSession.put>`, and :py:meth:`XNATSession.delete <xnat.XNATSession.delete>`.
    The methods do not query URIs but instead query XNATSession REST paths as described in the
    `XNATSession 1.6 REST API Directory <https://wiki.xnat.org/display/XNAT16/XNATSession+REST+API+Directory>`_.

    For an even lower level interfaces, the :py:attr:`XNATSession.interface <xnat.XNATSession.interface>`
    gives access to the underlying `requests <https://requests.readthedocs.org>`_ interface.
    This interface has the user credentials and benefits from the keep alive of this class.

    .. note:: XNATSession Objects have a client-side cache. This is for efficiency, but might cause
              problems if the server is being changed by a different client. It is possible
              to clear the current cache using :py:meth:`XNATSession.clearcache <xnat.XNATSession.clearcache>`.
              Turning off caching complete can be done by setting
              :py:attr:`XNATSession.caching <xnat.XNATSession.caching>`.

    .. warning:: You should NOT try use this class directly, it should only
                 be created by :py:func:`xnat.connect <xnat.connect>`.
    """

    # Class lookup to populate
    XNAT_CLASS_LOOKUP = {}

    def __init__(self, server, interface=None, user=None, password=None, keepalive=840, debug=False):
        self.classes = None
        self._interface = interface
        self._projects = None
        self._server = parse.urlparse(server) if server else None
        self._cache = {'__objects__': {}}
        self.caching = True
        self._source_code_file = None
        self._services = Services(xnat_session=self)
        self._prearchive = Prearchive(xnat_session=self)
        self._debug = debug
        self.inspect = Inspect(self)

        # Set the keep alive settings and spawn the keepalive thread for sending heartbeats
        if isinstance(keepalive, int) and keepalive > 0:
            self._keepalive = True
            self._keepalive_interval = keepalive
        else:
            self._keepalive = False
            self._keepalive_interval = 14 * 60

        self._keepalive_running = False
        self._keepalive_thread = None
        self._keepalive_event = threading.Event()

        # If needed connect here
        self.connect(server=server, user=user, password=password)

    def __del__(self):
        self.disconnect()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.disconnect()

    def connect(self, server=None, user=None, password=None):
        # If not connected, connect now
        if self.interface is None:
            if server is None:
                raise ValueError('Cannot connect if no server is given')
            print('[INFO] Connecting to server {}'.format(server))
            if self._interface is not None:
                self.disconnect()

            self._server = parse.urlparse(server)

            if user is None and password is None:
                print('[INFO] Retrieving login info for {}'.format(self._server.netloc))
                try:
                    user, _, password = netrc.netrc().authenticators(self._server.netloc)
                except TypeError:
                    raise ValueError('Could not retrieve login info for "{}" from the .netrc file!'.format(server))

            self._interface = requests.Session()
            if (user is not None) or (password is not None):
                self._interface.auth = (user, password)

        # Create a keepalive thread
        self._keepalive_running = True
        self._keepalive_thread = threading.Thread(target=self._keepalive_thread_run)
        self._keepalive_thread.daemon = True  # Make sure thread stops if program stops
        self._keepalive_thread.start()

    def disconnect(self):
        # Stop the keepalive thread
        self._keepalive_running = False
        self._keepalive_event.set()

        if self._keepalive_thread is not None:
            if self._keepalive_thread.is_alive():
                self._keepalive_thread.join(3.0)
            self._keepalive_thread = None

        # Kill the session
        if self._server is not None and self._interface is not None:
            self.delete('/data/JSESSION', headers={'Connection': 'close'})

        # Set the server and interface to None
        self._interface = None
        self._server = None

        # If this object is created using an automatically generated file
        # we have to remove it.
        if self._source_code_file is not None:
            source_pyc = self._source_code_file + 'c'
            if os.path.isfile(self._source_code_file):
                os.remove(self._source_code_file)
                self._source_code_file = None
            if os.path.isfile(source_pyc):
                os.remove(source_pyc)

        self.classes = None

    @property
    def keepalive(self):
        return self._keepalive

    @keepalive.setter
    def keepalive(self, value):
        if isinstance(value, int):
            if value > 0:
                self._keepalive_interval = value
                value = True
            else:
                value = False

        elif not isinstance(value, bool):
            raise TypeError('Type should be an integer or boolean!')

        self._keepalive = value

        if self.keepalive:
            # Send a new heartbeat and restart the timer to make sure the interval is correct
            self._keepalive_event.set()
            self.heartbeat()

    def heartbeat(self):
        self.get('/data/JSESSION')

    def _keepalive_thread_run(self):
        # This thread runs until the program stops, it should be inexpensive if not used due to the long sleep time
        while self._keepalive_running:
            # Wait returns False on timeout and True otherwise
            if not self._keepalive_event.wait(self._keepalive_interval):
                if self.keepalive:
                    self.heartbeat()
            else:
                self._keepalive_event.clear()

    @property
    def debug(self):
        return self._debug

    @property
    def interface(self):
        """
        The underlying `requests <https://requests.readthedocs.org>`_ interface used.
        """
        return self._interface

    @property
    def uri(self):
        return '/data/archive'

    @property
    def fulluri(self):
        return self.uri

    @property
    def xnat_session(self):
        return self

    def _check_response(self, response, accepted_status=None, uri=None):
        if self.debug:
            print('[DEBUG] Received response with status code: {}'.format(response.status_code))

        if accepted_status is None:
            accepted_status = [200, 201, 202, 203, 204, 205, 206]  # All successful responses of HTML
        if response.status_code not in accepted_status or response.text.startswith(('<!DOCTYPE', '<html>')):
            raise exceptions.XNATResponseError('Invalid response from XNATSession for url {} (status {}):\n{}'.format(uri, response.status_code, response.text))

    def get(self, path, format=None, query=None, accepted_status=None):
        """
        Retrieve the content of a given REST directory.

        :param str path: the path of the uri to retrieve (e.g. "/data/archive/projects")
                         the remained for the uri is constructed automatically
        :param str format: the format of the request, this will add the format= to the query string
        :param dict query: the values to be added to the query string in the uri
        :param list accepted_status: a list of the valid values for the return code, default [200]
        :returns: the requests reponse
        :rtype: requests.Response
        """
        accepted_status = accepted_status or [200]
        uri = self._format_uri(path, format, query=query)

        if self.debug:
            print('[DEBUG] GET URI {}'.format(uri))

        try:
            response = self.interface.get(uri)
        except requests.exceptions.SSLError:
            raise exceptions.XNATSSLError('Encountered a problem with the SSL connection, are you sure the server is offering https?')
        self._check_response(response, accepted_status=accepted_status, uri=uri)  # Allow OK, as we want to get data
        return response

    def head(self, path, accepted_status=None):
        """
        Retrieve the header for a http request of a given REST directory.

        :param str path: the path of the uri to retrieve (e.g. "/data/archive/projects")
                         the remained for the uri is constructed automatically
        :param list accepted_status: a list of the valid values for the return code, default [200]
        :returns: the requests reponse
        :rtype: requests.Response
        """
        accepted_status = accepted_status or [200]
        uri = self._format_uri(path)

        if self.debug:
            print('[DEBUG] GET URI {}'.format(uri))

        try:
            response = self.interface.head(uri)
        except requests.exceptions.SSLError:
            raise exceptions.XNATSSLError('Encountered a problem with the SSL connection, are you sure the server is offering https?')
        self._check_response(response, accepted_status=accepted_status, uri=uri)  # Allow OK, as we want to get data
        return response

    def post(self, path, data=None, json=None, format=None, query=None, accepted_status=None):
        """
        Post data to a given REST directory.

        :param str path: the path of the uri to retrieve (e.g. "/data/archive/projects")
                         the remained for the uri is constructed automatically
        :param data: Dictionary, bytes, or file-like object to send in the body of the :class:`Request`.
        :param json: json data to send in the body of the :class:`Request`.
        :param str format: the format of the request, this will add the format= to the query string
        :param dict query: the values to be added to the query string in the uri
        :param list accepted_status: a list of the valid values for the return code, default [200]
        :returns: the requests reponse
        :rtype: requests.Response
        """
        accepted_status = accepted_status or [200, 201]
        uri = self._format_uri(path, format, query=query)

        if self.debug:
            print('[DEBUG] POST URI {}'.format(uri))
            print('[DEBUG] POST DATA {}'.format(data))

        try:
            response = self._interface.post(uri, data=data, json=json)
        except requests.exceptions.SSLError:
            raise exceptions.XNATSSLError('Encountered a problem with the SSL connection, are you sure the server is offering https?')
        self._check_response(response, accepted_status=accepted_status, uri=uri)
        return response

    def put(self, path, data=None, files=None, json=None, format=None, query=None, accepted_status=None):
        """
        Put the content of a given REST directory.

        :param str path: the path of the uri to retrieve (e.g. "/data/archive/projects")
                         the remained for the uri is constructed automatically
        :param data: Dictionary, bytes, or file-like object to send in the body of the :class:`Request`.
        :param json: json data to send in the body of the :class:`Request`.
        :param files: Dictionary of ``'name': file-like-objects`` (or ``{'name': file-tuple}``) for multipart encoding upload.
                      ``file-tuple`` can be a 2-tuple ``('filename', fileobj)``, 3-tuple ``('filename', fileobj, 'content_type')``
                      or a 4-tuple ``('filename', fileobj, 'content_type', custom_headers)``, where ``'content-type'`` is a string
                      defining the content type of the given file and ``custom_headers`` a dict-like object containing additional headers
                      to add for the file.
        :param str format: the format of the request, this will add the format= to the query string
        :param dict query: the values to be added to the query string in the uri
        :param list accepted_status: a list of the valid values for the return code, default [200]
        :returns: the requests reponse
        :rtype: requests.Response
        """
        accepted_status = accepted_status or [200, 201]
        uri = self._format_uri(path, format, query=query)

        if self.debug:
            print('[DEBUG] PUT URI {}'.format(uri))
            print('[DEBUG] PUT DATA {}'.format(data))
            print('[DEBUG] PUT FILES {}'.format(data))

        try:
            response = self._interface.put(uri, data=data, files=files, json=json)
        except requests.exceptions.SSLError:
            raise exceptions.XNATSSLError('Encountered a problem with the SSL connection, are you sure the server is offering https?')
        self._check_response(response, accepted_status=accepted_status, uri=uri)  # Allow created OK or Create status (OK if already exists)
        return response

    def delete(self, path, headers=None, accepted_status=None, query=None):
        """
        Delete the content of a given REST directory.

        :param str path: the path of the uri to retrieve (e.g. "/data/archive/projects")
                         the remained for the uri is constructed automatically
        :param str format: the format of the request, this will add the format= to the query string
        :param dict query: the values to be added to the query string in the uri
        :param list accepted_status: a list of the valid values for the return code, default [200]
        :returns: the requests reponse
        :rtype: requests.Response
        """
        accepted_status = accepted_status or [200]
        uri = self._format_uri(path, query=query)

        if self.debug:
            print('[DEBUG] DELETE URI {}'.format(uri))
            print('[DEBUG] DELETE HEADERS {}'.format(headers))

        try:
            response = self.interface.delete(uri, headers=headers)
        except requests.exceptions.SSLError:
            raise exceptions.XNATSSLError('Encountered a problem with the SSL connection, are you sure the server is offering https?')
        self._check_response(response, accepted_status=accepted_status, uri=uri)
        return response

    def _format_uri(self, path, format=None, query=None):
        if path[0] != '/':
            raise ValueError('The requested URI path should start with a / (e.g. /data/projects), found {}'.format(path))

        if query is None:
            query = {}

        if format is not None:
            query['format'] = format

        # Create the query string
        if len(query) > 0:
            query_string = parse.urlencode(query)
        else:
            query_string = ''

        data = (self._server.scheme,
                self._server.netloc,
                self._server.path.rstrip('/') + path,
                '',
                query_string,
                '')

        return parse.urlunparse(data)

    def get_json(self, uri, query=None):
        """
        Helper function that perform a GET, but sets the format to JSON and
        parses the result as JSON

        :param str uri: the path of the uri to retrieve (e.g. "/data/archive/projects")
                         the remained for the uri is constructed automatically
        :param dict query: the values to be added to the query string in the uri
        """
        response = self.get(uri, format='json', query=query)
        try:
            return response.json()
        except ValueError:
            raise ValueError('Could not decode JSON from [{}] {}'.format(uri, response.text))

    def download_stream(self, uri, target_stream, format=None, verbose=False, chunk_size=524288):
        uri = self._format_uri(uri, format=format)
        if self.debug:
            print('[DEBUG] DOWNLOAD URI {}'.format(uri))

        # Stream the get and write to file
        response = self.interface.get(uri, stream=True)

        if response.status_code != 200:
            raise exceptions.XNATResponseError('Invalid response from XNATSession for url {} (status {}):\n{}'.format(uri, response.status_code, response.text))

        bytes_read = 0
        if verbose:
            print('Downloading {}:'.format(uri))
        for chunk in response.iter_content(chunk_size):
            if bytes_read == 0 and chunk[0] == '<' and chunk.startswith(('<!DOCTYPE', '<html>')):
                raise ValueError('Invalid response from XNATSession (status {}):\n{}'.format(response.status_code, chunk))

            bytes_read += len(chunk)
            target_stream.write(chunk)

            if verbose:
                sys.stdout.write('\r{:d} kb'.format(bytes_read / 1024))
                sys.stdout.flush()

    def download(self, uri, target, format=None, verbose=True):
        """
        Download uri to a target file
        """
        with open(target, 'wb') as out_fh:
            self.download_stream(uri, out_fh, format=format, verbose=verbose)

        if verbose:
            sys.stdout.write('\nSaved as {}...\n'.format(target))
            sys.stdout.flush()

    def download_zip(self, uri, target, verbose=True):
        """
        Download uri to a target zip file
        """
        self.download(uri, target, format='zip', verbose=verbose)

    def upload(self, uri, file_, retries=1, query=None, content_type=None, method='put'):
        uri = self._format_uri(uri, query=query)
        if self.debug:
            print('[DEBUG] UPLOAD URI {}'.format(uri))
        attempt = 0
        file_handle = None
        opened_file = False

        try:
            while attempt < retries:
                if isinstance(file_, file):
                    # File is open file handle, seek to 0
                    file_handle = file_
                    file_.seek(0)
                elif os.path.isfile(file_):
                    # File is str path to file
                    file_handle = open(file_, 'rb')
                    opened_file = True
                else:
                    # File is data to upload
                    file_handle = file_

                attempt += 1

                try:
                    # Set the content type header
                    if content_type is None:
                        headers = {'Content-Type': 'application/octet-stream'}
                    else:
                        headers = {'Content-Type': content_type}

                    if method == 'put':
                        response = self.interface.put(uri, data=file_handle, headers=headers)
                    elif method == 'post':
                        response = self.interface.post(uri, data=file_handle, headers=headers)
                    else:
                        raise ValueError('Invalid upload method "{}" should be either put or post.'.format(method))
                    self._check_response(response)
                    return response
                except exceptions.XNATResponseError:
                    pass
        finally:
            if opened_file:
                file_handle.close()

        # We didn't return correctly, so we have an error
        raise exceptions.XNATUploadError('Upload failed after {} attempts! Status code {}, response text {}'.format(retries, response.status_code, response.text))

    @property
    def scanners(self):
        """
        A list of scanners referenced in XNATSession
        """
        return [x['scanner'] for x in self.xnat_session.get_json('/data/archive/scanners')['ResultSet']['Result']]

    @property
    def scan_types(self):
        """
         A list of scan types associated with this XNATSession instance
        """
        return self.xnat_session.get_json('/data/archive/scan_types')['ResultSet']['Result']

    @property
    def xnat_version(self):
        """
        The version of the XNAT server
        """
        return self.get('/data/version').text

    def create_object(self, uri, type_=None, fieldname=None, **kwargs):
        if (uri, fieldname) not in self._cache['__objects__']:
            if type_ is None:
                if self.xnat_session.debug:
                    print('[DEBUG] Type unknown, fetching data to get type')
                data = self.xnat_session.get_json(uri)
                type_ = data['items'][0]['meta']['xsi:type']
                datafields = data['items'][0]['data_fields']
            else:
                datafields = None

            if self.xnat_session.debug:
                print('[DEBUG] Looking up type {} [{}]'.format(type_, type(type_).__name__))
            if type_ not in self.XNAT_CLASS_LOOKUP:
                raise KeyError('Type {} unknow to this XNATSession REST client (see XNAT_CLASS_LOOKUP class variable)'.format(type_))

            cls = self.XNAT_CLASS_LOOKUP[type_]

            if self.xnat_session.debug:
                print('[DEBUG] Creating object of type {}'.format(cls))

            self._cache['__objects__'][uri, fieldname] = cls(uri, self, datafields=datafields, fieldname=fieldname, **kwargs)
        elif self.debug:
            print('[DEBUG] Fetching object {} from cache'.format(uri))

        return self._cache['__objects__'][uri, fieldname]

    @property
    @caching
    def projects(self):
        """
        Listing of all projects on the XNAT server
        """
        return XNATListing(self.uri + '/projects',
                           xnat_session=self.xnat_session,
                           parent=self,
                           field_name='projects',
                           xsi_type='xnat:projectData',
                           secondary_lookup_field='name')

    @property
    @caching
    def subjects(self):
        """
        Listing of all subjects on the XNAT server
        """
        return XNATListing(self.uri + '/subjects',
                           xnat_session=self.xnat_session,
                           parent=self,
                           field_name='subjects',
                           xsi_type='xnat:subjectData',
                           secondary_lookup_field='label')

    @property
    @caching
    def experiments(self):
        """
        Listing of all experiments on the XNAT server
        """
        return XNATListing(self.uri + '/experiments',
                           xnat_session=self.xnat_session,
                           parent=self,
                           field_name='experiments',
                           secondary_lookup_field='label')

    @property
    def prearchive(self):
        """
        Representation of the prearchive on the XNAT server, see :py:mod:`xnat.prearchive`
        """
        return self._prearchive

    @property
    def services(self):
        """
        Collection of services, see :py:mod:`xnat.services`
        """
        return self._services

    def clearcache(self):
        """
        Clear the cache of the listings in the Session object
        """
        self._cache.clear()
        self._cache['__objects__'] = {}
