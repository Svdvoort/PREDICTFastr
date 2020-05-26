# PREDICT v3.1.4

## PREDICT: a Radiomics Extensive Digital Interchangable Classification Toolkit

This is an open-source python package supporting radiomics image feature extraction.

### Documentation

For more information, see the sphinx generated documentation available [here (WIP)](http://predict.readthedocs.io/).

Alternatively, you can generate the documentation by checking out the master branch and running from the root directory:

    python setup.py build_sphinx

The documentation can then be viewed in a browser by opening `PACKAGE_ROOT\build\sphinx\html\index.html`.

### Installation

PREDICT has currently been tested on Ubuntu 16.04 and 18.04, and Windows 10
using Python 3.6.6 and higher.

The package can be installed through pip :

    pip install PREDICT

Alternatively, you can use the provided setup.py file:

    python setup.py install

Make sure you first install the required packages:

    pip install -r requirements.txt


### 3rd-party packages used in PREDICT:
We mainly rely on the following packages:

 - SimpleITK (Image loading and preprocessing)
 - numpy (Feature computation)
 - scikit-image
 - pandas (Storage)
 - PyRadiomics
 - pydicom

See also the [requirements file](requirements.txt).

### License
This package is covered by the open source [APACHE 2.0 License](APACHE-LICENSE-2.0). When using PREDICT, please use the following DOI: [![DOI](https://zenodo.org/badge/92298822.svg)](https://zenodo.org/badge/latestdoi/92298822).

### Contact
We are happy to help you with any questions: please send us a message or create an issue on Github.
