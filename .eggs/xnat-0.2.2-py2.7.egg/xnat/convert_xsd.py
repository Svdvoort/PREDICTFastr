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
from __future__ import print_function
from __future__ import unicode_literals
import contextlib
import inspect
import keyword
import re
from xml.etree import ElementTree

from . import core
from . import xnatbases
from .constants import SECONDARY_LOOKUP_FIELDS, FIELD_HINTS


# TODO: Add more fields to FileData from [Name, Size, URI, cat_ID, collection, file_content, file_format, tile_tags]?
FILE_HEADER = '''
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
from xnat.core import XNATObject, XNATSubObject, XNATListing, caching
from xnat.utils import mixedproperty


SESSION = None


def current_session():
    return SESSION


# These mixins are to set the xnat_session automatically in all created classes
class XNATObjectMixin(XNATObject):
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
    SECONDARY_LOOKUP_FIELD = "{file_secondary_lookup}"
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
XNAT_CLASS_LOOKUP = {{
    "xnat:fileData": FileData,
}}


# The following code represents the data structure of the XNAT server
# It is automatically generated using
{schemas}


'''

# TODO: Add display identifiers support
# <xs:annotation>
# <xs:appinfo>
# <xdat:element displayIdentifiers="label"/>
# </xs:appinfo>
# <xs:documentation>An individual person involved in experimental research</xs:documentation>
# </xs:annotation>
# <xs:sequence>
# TODO: Add XPATHs for setting SubObjects
# TODO: Make Listings without key and with numeric index possible
# TODO: Fix scan parameters https://groups.google.com/forum/#!topic/xnat_discussion/GBZoamC2ZmY


class ClassRepresentation(object):
    # Override strings for certain properties
    SUBSTITUTIONS = {
            "fields": "    @property\n    def fields(self):\n        return self._fields",
            }

    # Fields for lookup besides the id

    def __init__(self, parser, name, xsi_type, base_class='XNATObjectMixin', parent=None, field_name=None):
        self.parser = parser
        self.name = name
        self._xsi_type = xsi_type
        self.baseclass = base_class
        self.properties = {}
        self.parent = parent
        self.field_name = field_name
        self.abstract = False

    def __repr__(self):
        return '<ClassRepresentation {}({})>'.format(self.name, self.baseclass)

    def __str__(self):
        base = self.get_base_template()
        if base is not None:
            base_source = inspect.getsource(base)
            base_source = re.sub(r'class {}\(XNATObject\):'.format(self.python_name), 'class {}({}):'.format(self.python_name, self.python_baseclass), base_source)
            header = base_source.strip() + '\n\n    # END HEADER\n'
        else:
            header = '# No base template found for {}\n'.format(self.python_name)
            header += "class {name}({base}):\n".format(name=self.python_name, base=self.python_baseclass)

        header += "    # Abstract: {}\n".format(self.abstract)

        if 'fields' in self.properties:
            header += "    _HAS_FIELDS = True\n"

        if self.parent is not None:
            header += "    _PARENT_CLASS = {}\n".format(self.python_parentclass)
            header += "    _FIELD_NAME = '{}'\n".format(self.field_name)
        elif self.xsi_type in FIELD_HINTS:
            header += "    _CONTAINED_IN = '{}'\n".format(FIELD_HINTS[self.xsi_type])

        header += "    _XSI_TYPE = '{}'\n\n".format(self.xsi_type)
        if self.xsi_type in SECONDARY_LOOKUP_FIELDS:
            header += self.init

        properties = [self.properties[k] for k in sorted(self.properties.keys())]

        properties = '\n\n'.join(self.print_property(p) for p in properties if not self.hasattr(p.clean_name))
        return '{}{}'.format(header, properties)

    @property
    def xsi_type(self):
        xsi_type_name, xsi_type_extension = self._xsi_type
        return self.parser.xsi_mapping.get(xsi_type_name, 'xnat:' + self.name) + xsi_type_extension

    def hasattr(self, name):
        base = self.get_base_template()

        if base is not None:
            return hasattr(base, name)
        else:
            base = self.parser.class_list.get(self.baseclass)
            if base is not None:
                return base.hasattr(name)
            else:
                base = self.get_super_class()
                return hasattr(base, name)

    @property
    def python_name(self):
        name = ''.join(x if x.isalnum() else '_' for x in self.name)
        name = re.sub('_+', '_', name)
        return name[0].upper() + name[1:]

    @property
    def python_baseclass(self):
        name = ''.join(x if x.isalnum() else '_' for x in self.baseclass)
        name = re.sub('_+', '_', name)
        return name[0].upper() + name[1:]

    @property
    def python_parentclass(self):
        name = ''.join(x if x.isalnum() else '_' for x in self.parent)
        name = re.sub('_+', '_', name)
        return name[0].upper() + name[1:]

    def get_base_template(self):
        if hasattr(xnatbases, self.python_name):
            return getattr(xnatbases, self.python_name)

    def get_super_class(self):
        if hasattr(core, self.python_baseclass):
            return getattr(core, self.python_baseclass)

    def print_property(self, prop):
        if prop.name in self.SUBSTITUTIONS:
            return self.SUBSTITUTIONS[prop.name]
        else:
            data = str(prop)
            if prop.name == SECONDARY_LOOKUP_FIELDS.get(self.name, '!None'):
                head, tail = data.split('\n', 1)
                data = '{}\n    @caching\n{}'.format(head, tail)
            return data

    @property
    def init(self):
        return \
"""    def __init__(self, uri=None, xnat_session=None, id_=None, datafields=None, parent=None, {lookup}=None, **kwargs):
        super({name}, self).__init__(uri=uri, xnat_session=xnat_session, id_=id_, datafields=datafields, parent=parent, {lookup}={lookup}, **kwargs)
        if {lookup} is not None:
            self._cache['{lookup}'] = {lookup}

""".format(name=self.python_name, lookup=SECONDARY_LOOKUP_FIELDS[self.xsi_type])


class PropertyRepresentation(object):
    def __init__(self, parser, name, type_=None):
        self.parser = parser
        self.name = name
        self.restrictions = {}
        self.type_ = type_
        self.docstring = None
        self.is_listing = False

    def __repr__(self):
        return '<PropertyRepresentation {}({})>'.format(self.name, self.type_)

    def __str__(self):
        docstring = '\n        """ {} """'.format(self.docstring) if self.docstring is not None else ''
        if self.is_listing:
            return """    @property
    @caching
    def {clean_name}(self):
        # Generate automatically, type: {type_} (listing {is_listing})
        return XNATListing(self.fulluri + '/{name}',
                           xnat_session=self.xnat_session,
                           parent=self,
                           field_name='{name}',
                           xsi_type='{type_}')""".format(clean_name=self.clean_name,
                                                         name=self.name,
                                                         type_=self.type_,
                                                         is_listing=self.is_listing)
        elif not (self.type_ is None or self.type_.startswith('xnat:')):
            return \
        """    @mixedproperty
    def {clean_name}(cls):{docstring}
        # Generate automatically, type: {type_} (listing {is_listing})
        return search.SearchField(cls, "{name}")

    @{clean_name}.getter
    def {clean_name}(self):
        # Generate automatically, type: {type_}
        return self.get("{name}", type_="{type_}")

    @ {clean_name}.setter
    def {clean_name}(self, value):{docstring}{restrictions}
        # Generate automatically, type: {type_}
        self.set("{name}", value, type_="{type_}")""".format(clean_name=self.clean_name,
                                                             docstring=docstring,
                                                             name=self.name,
                                                             type_=self.type_,
                                                             is_listing=self.is_listing,
                                                             restrictions=self.restrictions_code())
        elif self.type_ is None:
            xsi_type = "'{{}}/{{}}'.format(cls._XSI_TYPE, '{}')".format(self.name)
            return \
        """    @mixedproperty
    def {clean_name}(cls):{docstring}
        # Generate automatically, type: {type_} (listing {is_listing})
        return XNAT_CLASS_LOOKUP["{xsi_type}"]

    @{clean_name}.getter
    @caching
    def {clean_name}(self):
        # Generated automatically, type: {type_}
        return self.get_object("{name}", {xsi_type})""".format(clean_name=self.clean_name,
                                                               docstring=docstring,
                                                               name=self.name,
                                                               type_=self.type_,
                                                               is_listing=self.is_listing,
                                                               xsi_type=xsi_type)
        else:
            xsi_type = core.TYPE_HINTS.get(self.name, self.type_)

            return \
        """    @mixedproperty
    def {clean_name}(cls):{docstring}
        # Generate automatically, type: {type_} (listing {is_listing})
        return XNAT_CLASS_LOOKUP["{xsi_type}"]

    @{clean_name}.getter
    @caching
    def {clean_name}(self):
        # Generated automatically, type: {type_}
        return self.get_object("{name}")""".format(clean_name=self.clean_name,
                                                   docstring=docstring,
                                                   name=self.name,
                                                   type_=self.type_,
                                                   is_listing=self.is_listing,
                                                   xsi_type=xsi_type)

    @property
    def clean_name(self):
        name = re.sub('[^0-9a-zA-Z]+', '_', self.name)

        if keyword.iskeyword(name):
            name += '_'
        return name.lower()

    def restrictions_code(self):
        if len(self.restrictions) > 0:
            data = '\n        # Restrictions for value'
            if 'min' in self.restrictions:
                data += "\n        if value < {min}:\n            raise ValueError('{name} has to be greater than or equal to {min}')\n".format(name=self.name, min=self.restrictions['min'])
            if 'max' in self.restrictions:
                data += "\n        if value > {max}:\n            raise ValueError('{name} has to be smaller than or equal to {max}')\n".format(name=self.name, max=self.restrictions['max'])
            if 'maxlength' in self.restrictions:
                data += "\n        if len(value) > {maxlength}:\n            raise ValueError('length {name} has to be smaller than or equal to {maxlength}')\n".format(name=self.name, maxlength=self.restrictions['maxlength'])
            if 'minlength' in self.restrictions:
                data += "\n        if len(value) < {minlength}:\n            raise ValueError('length {name} has to be larger than or equal to {minlength}')\n".format(name=self.name, minlength=self.restrictions['minlength'])
            if 'enum' in self.restrictions:
                data += "\n        if value not in [{enum}]:\n            raise ValueError('{name} has to be one of: {enum}')\n".format(name=self.name, enum=', '.join('"{}"'.format(x.replace("'", "\\'")) for x in self.restrictions['enum']))

            return data
        else:
            return ''


class SchemaParser(object):
    def __init__(self, debug=False):
        self.class_list = {}
        self.unknown_tags = set()
        self.new_class_stack = [None]
        self.new_property_stack = [None]
        self.property_prefixes = []
        self.debug = debug
        self.schemas = []
        self.xsi_mapping = {}

    def parse_schema_uri(self, requests_session, schema_uri):
        print('[INFO] Retrieving schema from {}'.format(schema_uri))

        if self.debug:
            print('[DEBUG] GET SCHEMA {}'.format(schema_uri))
        resp = requests_session.get(schema_uri, headers={'Accept-Encoding': None})
        data = resp.text

        try:
            root = ElementTree.fromstring(data)
        except ElementTree.ParseError as exception:
            if 'action="/j_spring_security_check"' in data:
                print('[ERROR] You do not have access to this XNAT server, please check your credentials!')
            elif 'java.lang.IllegalStateException' in data:
                print('[ERROR] The server returned an error. You probably do not'
                      ' have access to this XNAT server, please check your credentials!')
            else:
                print('[ERROR] Could not parse schema from {}, not valid XML found'.format(schema_uri))

                if self.debug:
                    print('[DEBUG] XML schema request returned the following response: [{}] {}'.format(resp.status_code,
                                                                                                       data))
            return False

        # Register schema as being loaded
        self.schemas.append(schema_uri)

        # Parse xml schema
        self.parse(root, toplevel=True)

        if self.debug:
            print('[DEBUG] Found {} unknown tags: {}'.format(len(self.unknown_tags),
                                                             self.unknown_tags))

        return True

    @staticmethod
    def find_schema_uris(text):
        try:
            root = ElementTree.fromstring(text)
        except ElementTree.ParseError:
            raise ValueError('Could not parse xml file')

        schemas_string = root.attrib.get('{http://www.w3.org/2001/XMLSchema-instance}schemaLocation', '')
        schemas = [x for x in schemas_string.split() if x.endswith('.xsd')]

        return schemas

    def __iter__(self):
        visited = set(['XNATObjectMixin', 'XNATSubObjectMixin'])
        tries = 0
        yielded_anything = True
        while len(visited) < len(self.class_list) and yielded_anything and tries < 250:
            yielded_anything = False
            for key, value in self.class_list.items():
                if key in visited:
                    continue

                base = value.baseclass
                if not base.startswith('xs:') and base not in visited:
                    continue

                if value.parent is not None and value.parent not in visited:
                    continue

                visited.add(key)
                yielded_anything = True
                yield value

            tries += 1

        if self.debug and len(visited) < len(self.class_list):
            print('[DEBUG] Visited: {}, expected: {}'.format(len(visited), len(self.class_list)))
            print('[DEBUG] Missed: {}'.format(set(self.class_list.keys()) - visited))
            print('[DEBUG] Spent {} iterations'.format(tries))

    @contextlib.contextmanager
    def descend(self, new_class=None, new_property=None, property_prefix=None):
        if new_class is not None:
            self.new_class_stack.append(new_class)
        if new_property is not None:
            self.new_property_stack.append(new_property)
        if property_prefix is not None:
            self.property_prefixes.append(property_prefix)

        yield

        if new_class is not None:
            self.new_class_stack.pop()
        if new_property is not None:
            self.new_property_stack.pop()
        if property_prefix is not None:
            self.property_prefixes.pop()

    @property
    def current_class(self):
        return self.new_class_stack[-1]

    @property
    def current_property(self):
        return self.new_property_stack[-1]

    def parse(self, element, toplevel=False):
        if toplevel:
            if element.tag != '{http://www.w3.org/2001/XMLSchema}schema':
                raise ValueError('File should contain a schema as root element!')

            for child in element.getchildren():
                if child.tag == '{http://www.w3.org/2001/XMLSchema}complexType':
                    self.parse(child)
                elif child.tag == '{http://www.w3.org/2001/XMLSchema}element':
                    name = child.get('name')
                    type_ = child.get('type')

                    if self.debug:
                        print('[DEBUG] Adding {} -> {} to XSI map'.format(name, type_))
                    self.xsi_mapping[name] = type_
                else:
                    if self.debug:
                        print('[DEBUG] skipping non-class top-level tag {}'.format(child.tag))

        else:
            if element.tag in self.PARSERS:
                self.PARSERS[element.tag](self, element)
            else:
                self.parse_unknown(element)

    # TODO: We should check the following restrictions: http://www.w3schools.com/xml/schema_facets.asp

    def parse_all(self, element):
        self.parse_children(element)

    def parse_annotation(self, element):
        self.parse_children(element)

    def parse_attribute(self, element):
        name = element.get('name')
        type_ = element.get('type')

        if self.current_class is not None:
            if name is None:
                if self.debug:
                    print('[DEBUG] Encountered attribute without name')
                return
            new_property = PropertyRepresentation(self, name, type_)
            self.current_class.properties[name] = new_property

            with self.descend(new_property=new_property):
                self.parse_children(element)

    def parse_children(self, element):
        for child in element.getchildren():
            self.parse(child)

    def parse_choice(self, element):
        self.parse_children(element)

    def parse_complex_content(self, element):
        self.parse_children(element)

    def parse_complex_type(self, element):
        name = element.get('name')
        xsi_type = name, ''
        base_class = 'XNATObjectMixin'
        parent = None
        field_name = None

        if name is None:
            name = self.current_class.name + self.current_property.name.capitalize()
            xsi_type = self.current_class._xsi_type[0], '{}/{}'.format(self.current_class._xsi_type[1],
                                                                       self.current_property.name)
            base_class = 'XNATSubObjectMixin'
            parent = self.current_class.name
            field_name = self.current_property.name

        new_class = ClassRepresentation(self,
                                        name=name,
                                        xsi_type=xsi_type,
                                        base_class=base_class,
                                        parent=parent,
                                        field_name=field_name)
        self.class_list[name] = new_class

        # Descend
        with self.descend(new_class=new_class):
            self.parse_children(element)

    def parse_documentation(self, element):
        if self.current_property is not None:
            self.current_property.docstring = element.text

    def parse_element(self, element):
        name = element.get('name')
        type_ = element.get('type')

        if name is None:
            abstract = element.get('abstract')
            if abstract is not None:
                self.current_class.abstract = abstract == "true"
            else:
                if self.debug:
                    print('[DEBUG] Encountered attribute without name')
            return

        if element.get('maxOccurs') == 'unbounded':
            if self.current_property is None:
                if self.debug:
                    print('[DEBUG] Listing without parent property: {} ({})'.format(name, type_))
            else:
                self.current_property.is_listing = True
                self.parse_children(element)
                if type_ is not None:
                    self.current_property.type_ = type_
        elif self.current_class is not None:
            if self.debug:
                print('[DEBUG] Found property {} ({})'.format(name, type_))
            new_property = PropertyRepresentation(self, name, type_)
            self.current_class.properties[name] = new_property

            with self.descend(new_property=new_property):
                self.parse_children(element)
        else:
            if self.debug:
                print('[DEBUG] Found XSI_MAPPING {} -> {}'.format(name, type_))
            # Top level element is xsi mapping
            self.xsi_mapping[name] = type_

    def parse_enumeration(self, element):
        if 'enum' in self.current_property.restrictions:
            self.current_property.restrictions['enum'].append(element.get('value'))
        else:
            self.current_property.restrictions['enum'] = [element.get('value')]

    def parse_error(self, element):
        raise NotImplementedError('The parser for {} has not yet been implemented'.format(element.tag))

    def parse_extension(self, element):
        old_base = self.current_class.baseclass
        new_base = element.get('base')
        if new_base.startswith('xnat:'):
            new_base = new_base[5:]
        if old_base in ['XNATObjectMixin', 'XNATSubObjectMixin']:
            self.current_class.baseclass = new_base
        else:
            raise ValueError('Trying to reset base class again from {} to {}'.format(old_base, new_base))

        self.parse_children(element)

    def parse_ignore(self, element):
        pass

    def parse_max_inclusive(self, element):
        self.current_property.restrictions['max'] = element.get('value')

    def parse_max_length(self, element):
        self.current_property.restrictions['maxlength'] = element.get('value')

    def parse_min_inclusive(self, element):
        self.current_property.restrictions['min'] = element.get('value')

    def parse_min_length(self, element):
        self.current_property.restrictions['minlength'] = element.get('value')

    def parse_restriction(self, element):
        old_type = self.current_property.type_
        new_type = element.get('base')

        if old_type is not None:
            raise ValueError('Trying to override a type from a restriction!? (from {} to {})'.format(old_type, new_type))

        self.current_property.type_ = new_type

        self.parse_children(element)

    def parse_schema(self, element):
        self.parse_children(element)

    def parse_sequence(self, element):
        self.parse_children(element)

    def parse_simple_content(self, element):
        self.parse_children(element)

    def parse_simple_type(self, element):
        self.parse_children(element)

    def parse_unknown(self, element):
        self.unknown_tags.add(element.tag)

    def parse_xdat_element(self, element):
        abstract = element.get("abstract")
        if abstract is not None:
            self.current_class.abstract = abstract == "true"

    PARSERS = {
        '{http://www.w3.org/2001/XMLSchema}all': parse_all,
        '{http://www.w3.org/2001/XMLSchema}annotation': parse_annotation,
        '{http://www.w3.org/2001/XMLSchema}appinfo': parse_children,
        '{http://www.w3.org/2001/XMLSchema}attribute': parse_attribute,
        '{http://www.w3.org/2001/XMLSchema}attributeGroup': parse_error,
        '{http://www.w3.org/2001/XMLSchema}choice': parse_choice,
        '{http://www.w3.org/2001/XMLSchema}complexContent': parse_complex_content,
        '{http://www.w3.org/2001/XMLSchema}complexType': parse_complex_type,
        '{http://www.w3.org/2001/XMLSchema}documentation': parse_documentation,
        '{http://www.w3.org/2001/XMLSchema}element': parse_element,
        '{http://www.w3.org/2001/XMLSchema}enumeration': parse_enumeration,
        '{http://www.w3.org/2001/XMLSchema}extension': parse_extension,
        '{http://www.w3.org/2001/XMLSchema}import': parse_ignore,
        '{http://www.w3.org/2001/XMLSchema}group': parse_error,
        '{http://www.w3.org/2001/XMLSchema}maxInclusive': parse_max_inclusive,
        '{http://www.w3.org/2001/XMLSchema}maxLength': parse_max_length,
        '{http://www.w3.org/2001/XMLSchema}minInclusive': parse_min_inclusive,
        '{http://www.w3.org/2001/XMLSchema}minLength': parse_min_length,
        '{http://www.w3.org/2001/XMLSchema}restriction': parse_restriction,
        '{http://www.w3.org/2001/XMLSchema}schema': parse_schema,
        '{http://www.w3.org/2001/XMLSchema}sequence': parse_sequence,
        '{http://www.w3.org/2001/XMLSchema}simpleContent': parse_simple_content,
        '{http://www.w3.org/2001/XMLSchema}simpleType': parse_simple_type,
        '{http://nrg.wustl.edu/xdat}element': parse_xdat_element,
    }

    def write(self, code_file):
        schemas = '\n'.join('# - {}'.format(s) for s in self.schemas)
        code_file.write(FILE_HEADER.format(schemas=schemas,
                                           file_secondary_lookup=SECONDARY_LOOKUP_FIELDS['xnat:fileData']))
        code_file.write('\n\n\n'.join(str(c).strip() for c in self if not c.baseclass.startswith('xs:') and c.name is not None))
