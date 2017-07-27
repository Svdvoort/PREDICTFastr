# Copyright 2011-2014 Biomedical Imaging Group Rotterdam, Departments of
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

"""
This package contains the entire client. The connect function is the only
function actually in the package. All following classes are created based on
the https://central.xnat.org/schema/xnat/xnat.xsd schema and the xnatcore and
xnatbase modules, using the convert_xsd.
"""

from __future__ import absolute_import
from __future__ import unicode_literals
import getpass
import hashlib
import imp
import os
import netrc
import tempfile
import time

import requests
from six.moves.urllib import parse

from .session import XNATSession
from .convert_xsd import SchemaParser

GEN_MODULES = {}

__all__ = ['connect']


def parse_schemas_16(parser, requests_session, server, extension_types=True, debug=False):
    # Retrieve schema from XNAT server
    schema_uri = '{}/schemas/xnat/xnat.xsd'.format(server.rstrip('/'))

    success = parser.parse_schema_uri(requests_session=requests_session,
                                      schema_uri=schema_uri)

    if not success:
        raise RuntimeError('Could not parse the xnat.xsd! See error log for details!')

    # Parse extension types
    if extension_types:
        projects_uri = '{}/data/projects?format=json'.format(server.rstrip('/'))
        response = requests_session.get(projects_uri)
        if response.status_code != 200:
            raise ValueError('Could not get project list from {} (status {})'.format(projects_uri,
                                                                                     response.status_code))
        try:
            project_id = response.json()['ResultSet']['Result'][0]['ID']
        except (KeyError, IndexError):
            raise ValueError('Could not find an example project for scanning extension types!')

        project_uri = '{}/data/projects/{}?format=xml'.format(server.rstrip('/'), project_id)
        response = requests_session.get(project_uri)

        if response.status_code != 200:
            raise ValueError('Could not get example project from {} (status {})'.format(project_uri,
                                                                                        response.status_code))

        schemas = parser.find_schema_uris(response.text)
        if schema_uri in schemas:
            if debug:
                print('[DEBUG] Removing schema {} from list'.format(schema_uri))
            schemas.remove(schema_uri)
        print('[INFO] Found additional schemas: {}'.format(schemas))

        for schema in schemas:
            parser.parse_schema_uri(requests_session=requests_session,
                                    schema_uri=schema)


def parse_schemas_17(parser, requests_session, server, debug=False):
    schemas_uri  = '{}/xapi/schemas'.format(server.rstrip('/'))
    schemas_request = requests_session.get(schemas_uri)

    if schemas_request.status_code != 200:
        print('[ERROR] Problem retrieving schemas list: [{}] {}'.format(schemas_request.status_code, schemas_request.text))
        raise ValueError('Problem retrieving schemas list: [{}] {}'.format(schemas_request.status_code, schemas_request.text))

    schema_list = schemas_request.json()
    schema_list = ['{server}/xapi/schemas/{schema}'.format(server=server.rstrip('/'), schema=x) for x in schema_list]
    
    for schema in schema_list:
        parser.parse_schema_uri(requests_session=requests_session,
                                schema_uri=schema)


def connect(server, user=None, password=None, verify=True, netrc_file=None, debug=False, extension_types=True):
    """
    Connect to a server and generate the correct classed based on the servers xnat.xsd
    This function returns an object that can be used as a context operator. It will call
    disconnect automatically when the context is left. If it is used as a function, then
    the user should call ``.disconnect()`` to destroy the session and temporary code file.

    :param str server: uri of the server to connect to (including http:// or https://)
    :param str user: username to use, leave empty to use netrc entry or anonymous login.
    :param str password: password to use with the username, leave empty when using netrc.
                         If a username is given and no password, there will be a prompt
                         on the console requesting the password.
    :param bool verify: verify the https certificates, if this is false the connection will
                        be encrypted with ssl, but the certificates are not checked. This is
                        potentially dangerous, but required for self-signed certificates.
    :param str netrc_file: alternative location to use for the netrc file (path pointing to
                           a file following the netrc syntax)
    :param debug bool: Set debug information printing on
    :return: XNAT session object
    :rtype: XNATSession

    Preferred use::

        >>> import xnat
        >>> with xnat.connect('https://central.xnat.org') as session:
        ...    subjects = session.projects['Sample_DICOM'].subjects
        ...    print('Subjects in the SampleDICOM project: {}'.format(subjects))
        Subjects in the SampleDICOM project: <XNATListing (CENTRAL_S01894, dcmtest1): <SubjectData CENTRAL_S01894>, (CENTRAL_S00461, PACE_HF_SUPINE): <SubjectData CENTRAL_S00461>>

    Alternative use::

        >>> import xnat
        >>> session = xnat.connect('https://central.xnat.org')
        >>> subjects = session.projects['Sample_DICOM'].subjects
        >>> print('Subjects in the SampleDICOM project: {}'.format(subjects))
        Subjects in the SampleDICOM project: <XNATListing (CENTRAL_S01894, dcmtest1): <SubjectData CENTRAL_S01894>, (CENTRAL_S00461, PACE_HF_SUPINE): <SubjectData CENTRAL_S00461>>
        >>> session.disconnect()
    """
    # Get the login info
    parsed_server = parse.urlparse(server)

    if user is None and password is None:
        print('[INFO] Retrieving login info for {}'.format(parsed_server.netloc))
        try:
            if netrc_file is None:
                netrc_file = os.path.join('~', '_netrc' if os.name == 'nt' else '.netrc')
                netrc_file = os.path.expanduser(netrc_file)
            user, _, password = netrc.netrc(netrc_file).authenticators(parsed_server.netloc)
        except (TypeError, IOError):
            print('[INFO] Could not find login for {}, continuing without login'.format(parsed_server.netloc))

    if user is not None and password is None:
        password = getpass.getpass(prompt="Please enter the password for user '{}':".format(user))

    # Create the correct requests session
    requests_session = requests.Session()

    if user is not None:
        requests_session.auth = (user, password)

    if not verify:
        requests_session.verify = False

    # Generate module
    parser = SchemaParser(debug=debug)

    # Parse schemas
    version_uri = '{}/data/version'.format(server.rstrip('/'))
    version_request = requests_session.get(version_uri)
    if version_request.status_code == 200:
        version = version_request.text
    else:
        schemas_uri  = '{}/xapi/schemas'.format(server.rstrip('/'))
        schemas_request = requests_session.get(schemas_uri)

        if schemas_request.status_code == 200:
            version = '1.7.0'
        else:
            print('[ERROR] Could not retrieve version: [{}] {}'.format(version_request.status_code, version_request.text))
            raise ValueError('Cannot continue on unknown XNAT version')

    if version.startswith('1.6'):
        print('[INFO] Found an 1.6 version ({})'.format(version))
        parse_schemas_16(parser, requests_session, server, extension_types=extension_types, debug=debug)
    elif version.startswith('1.7'):
        print('[INFO] Found an 1.7 version ({})'.format(version))
        parse_schemas_17(parser, requests_session, server, debug=debug)
    else:
        print('[ERROR] Found an unsupported version ({})'.format(version))
        raise ValueError('Cannot continue on unsupported XNAT version')

    # Write code to temp file
    with tempfile.NamedTemporaryFile(mode='w', suffix='_generated_xnat.py', delete=False) as code_file:
        parser.write(code_file=code_file)

    if debug:
        print('[DEBUG] Code file written to: {}'.format(code_file.name))

    # Import temp file as a module
    hasher = hashlib.md5()
    hasher.update(server.encode('utf-8'))
    hasher.update(str(time.time()).encode('utf-8'))

    # The module is loaded in its private namespace based on the code_file name
    xnat_module = imp.load_source('xnat_gen_{}'.format(hasher.hexdigest()),
                                  code_file.name)
    xnat_module._SOURCE_CODE_FILE = code_file.name

    if debug:
        print('[DEBUG] Loaded generated module')

    # Register all types parsed
    for cls in parser:
        if not (cls.name is None or cls.baseclass.startswith('xs:')):
            xnat_module.XNAT_CLASS_LOOKUP[cls.xsi_type] = getattr(xnat_module, cls.python_name)

    # Create the XNAT connection
    session = XNATSession(server=server, interface=requests_session, debug=debug)

    # FIXME: is this a good idea, it makes things simple, but I suppose we
    # FIXME: can no longer re-use the modules between sessions?
    xnat_module.SESSION = session

    # Add the required information from the module into the session object
    session.XNAT_CLASS_LOOKUP.update(xnat_module.XNAT_CLASS_LOOKUP)
    session.classes = xnat_module
    session._source_code_file = xnat_module._SOURCE_CODE_FILE

    return session
