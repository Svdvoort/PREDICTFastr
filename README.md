# PREDICT v1.0.0

## PREDICT: a Radiomics Extensive Differentiable Interchangable Classification Toolkit

This is an open-source python package supporting Radiomics medical image feature extraction and classification.

We aim to add a wide variety of features and classifiers to address a wide variety classification problems.
Through a modular setup, these can easily be interchanged and compared.


### Documentation

For more information, see the sphinx generated documentation available [here](http://predict.readthedocs.io/).

Alternatively, you can generate the documentation by checking out the master branch and running from the root directory:

    python setup.py build_sphinx

The documentation can then be viewed in a browser by opening `PACKAGE_ROOT\build\sphinx\html\index.html`.

### Installation

PREDICT has currently only been tested on Unix with Python 2.7.
The package can be installed through pip:

      pip install PREDICT

### 3rd-party packages used in PREDICT:
We mainly rely on the following packages:

 - SimpleITK (Image loading and preprocessing)
 - numpy (Feature computation)
 - sklearn, scipy (Classification)
 - FASTR (Fast and parallel workflow execution)
 - pandas (Storage)

See also the [requirements file](requirements.txt).

### WIP
- We are working on improving the documentation.
- We are working on the addition of different classifiers.
- Examples and unit tests will be added.

### License
This package is covered by the open source [APACHE 2.0 License](APACHE-LICENSE-2.0).

### Contact
We are happy to help you with any questions: please send us a message or create an issue.

We welcome contributions to PREDICT. We will soon make some guidelines.
