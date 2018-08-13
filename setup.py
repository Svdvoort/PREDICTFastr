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

import os
import sys
import subprocess
from setuptools.command.install import install
from setuptools.command.test import test as TestCommand
from setuptools import setup, find_packages

with open('requirements.txt', 'r') as fh:
    _requires = fh.read().splitlines()

with open('README.rst', 'r') as fh:
    _description = fh.read()

with open('version', 'r') as fh:
    __version__ = fh.read().splitlines()[0]

with open('test_requirements.txt', 'r') as fh:
    _tests_require = fh.read().splitlines()

with open('requirements-setup.txt', 'r') as fp:
    setup_requirements = list(filter(bool, (line.strip() for line in fp)))


def scan_dir(path, prefix=None):
    if prefix is None:
        prefix = path

    # Scan resources package for files to include
    file_list = []
    for root, dirs, files in os.walk(path):
        # Strip this part as setup wants relative directories
        root = root.replace(prefix, '')
        root = root.lstrip('/\\')

        for filename in files:
            if filename[0:8] == '__init__':
                continue
            file_list.append(os.path.join(root, filename))

    return file_list


# Determine the extra resources and scripts to pack
print('[setup.py] called with: {}'.format(' '.join(sys.argv)))
if hasattr(sys, 'real_prefix'):
    print('[setup.py] Installing in virtual env {} (real prefix: {})'.format(sys.prefix, sys.real_prefix))
else:
    print('[setup.py] Not inside a virtual env!')


# Set the entry point
entry_points = {
    "console_scripts": [
        "PREDICT = PREDICT.PREDICT:main",
    ]
}

# Determine the fastr config path
USER_DIR = os.path.expanduser(os.path.join('~', '.fastr'))
config_d = os.path.join(USER_DIR, 'config.d')


class NoseTestCommand(TestCommand):
    def finalize_options(self):
        TestCommand.finalize_options(self)
        self.test_args = []
        self.test_suite = True

    def run_tests(self):
        # Run nose ensuring that argv simulates running nosetests directly
        import nose
        nose.run_exit(argv=['nosetests'])


here = os.path.abspath(os.path.dirname(__file__))

# Numpy and scipy give an error: workaround is installing them first


class MyInstall(install):
    def run(self):
        try:
            # note cwd - this makes the current directory
            # the one with the Makefile.
            # subprocess.call(['pip install -r requirements-setup.txt'])
            # for i in setup_requirements:
            # regcommand = ('pip install ' + i)
            # regcommand = 'pip install -r requirements.txt'
            # print regcommand
            # proc = subprocess.Popen(regcommand,
            #                         shell=True,
            #                         stdin=subprocess.PIPE,
            #                         stdout=subprocess.PIPE,
            #                         stderr=subprocess.STDOUT,
            #                         )
            # stdout_value, stderr_value = proc.communicate('through stdin to stdout\n')

            # Install pyradiomics
            commands = 'git clone https://github.com/Radiomics/pyradiomics; cd pyradiomics; pip install -r requirements.txt; python setup.py -q install; cd ..; rm -r pyradiomics;'
            print(commands)
            proc = subprocess.Popen(commands,
                                    shell=True,
                                    stdin=subprocess.PIPE,
                                    stdout=subprocess.PIPE,
                                    stderr=subprocess.STDOUT,
                                    )
            stdout_value, stderr_value = proc.communicate('through stdin to stdout\n')

        except Exception as e:
            print(e)
            exit(1)
        else:
            install.run(self)


setup(
    name='PREDICT',
    version='2.1.0',
    description='Predict: a Radiomics Extensive D.... Interchangable Classification Toolkit.',
    long_description=_description,
    url='https://github.com/Svdvoort/PREDICTFastr',
    author='S. van der Voort, M. Starmans',
    author_email='s.vandervoort@erasmusmc.nl, m.starmans@erasmusmc.nl',
    license='Apache License, Version 2.0',
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'Intended Audience :: Healthcare Industry',
        'Intended Audience :: Information Technology',
        'Intended Audience :: Education',
        'License :: OSI Approved :: Apache Software License',
        'Natural Language :: English',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 2.7',
        'Topic :: Scientific/Engineering :: Information Analysis',
        'Topic :: System :: Distributed Computing',
        'Topic :: System :: Logging',
        'Topic :: Utilities',
    ],
    keywords='bioinformatics radiomics features',
    packages=find_packages(exclude=['build', '_docs', 'templates']),
    include_package_data=True,
    package_data={'PREDICT': ['versioninfo']},
    install_requires=_requires,
    tests_require=_tests_require,
    test_suite='nose.collector',
    cmdclass={'test': NoseTestCommand, 'install': MyInstall},
    entry_points=entry_points,
    data_files=[(config_d, ['PREDICT/fastrconfig/PREDICT_config.py'])]
    # setup_requires=_requires
)
