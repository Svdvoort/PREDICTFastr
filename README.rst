PREDICT v3.1.14
===============

PREDICT: a Radiomics Extensive Digital Interchangable Classification Toolkit
----------------------------------------------------------------------------

This is an open-source python package supporting radiomics image feature
extraction.

Documentation
~~~~~~~~~~~~~

For more information, see the sphinx generated documentation available
in the docs folder. PREDICT is mostly used through the WORC toolbox, in
which further documentation on the features computed is also available,
see https://worc.readthedocs.io/en/latest/static/features.html.

Alternatively, you can generate the documentation by checking out the
master branch and running from the root directory:

::

    python setup.py build_sphinx

The documentation can then be viewed in a browser by opening
``PACKAGE_ROOT\build\sphinx\html\index.html``.

Installation
~~~~~~~~~~~~

PREDICT has currently been tested on Ubuntu 16.04 and 18.04, and Windows
10 using Python 3.6.6 and higher.

The package can be installed through pip :

::

    pip install PREDICT

Alternatively, you can use the provided setup.py file:

::

    python setup.py install

Make sure you first install the required packages:

::

    pip install -r requirements.txt

3rd-party packages used in PREDICT:
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

We mainly rely on the following packages:

-  SimpleITK (Image loading and preprocessing)
-  numpy (Feature computation)
-  scikit-image
-  pandas (Storage)
-  PyRadiomics
-  pydicom

See also the `requirements file <requirements.txt>`__.

License
~~~~~~~

This package is covered by the open source `APACHE 2.0
License <APACHE-LICENSE-2.0>`__. When using PREDICT, please use the
following DOI: |DOI|.

Contact
~~~~~~~

We are happy to help you with any questions: please send us a message or
create an issue on Github.

.. |DOI| image:: https://zenodo.org/badge/doi/10.5281/zenodo.3854839.svg
   :target: https://zenodo.org/badge/latestdoi/92298822
