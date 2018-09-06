PREDICT v2.1.0
==============

PREDICT: a Radiomics Extensive Differentiable Interchangable Classification Toolkit
-----------------------------------------------------------------------------------

This is an open-source python package supporting Radiomics medical image
feature extraction and classification.

We aim to add a wide variety of features and classifiers to address a
wide variety classification problems. Through a modular setup, these can
easily be interchanged and compared.

Documentation
~~~~~~~~~~~~~

For more information, see the sphinx generated documentation available
`here (WIP) <http://predict.readthedocs.io/>`__.

Alternatively, you can generate the documentation by checking out the
master branch and running from the root directory:

::

    python setup.py build_sphinx

The documentation can then be viewed in a browser by opening
``PACKAGE_ROOT\build\sphinx\html\index.html``.

Installation
~~~~~~~~~~~~

PREDICT has currently only been tested on Unix with Python 2.7. The
package can be installed through pip :

::

    pip install PREDICT

Alternatively, you can use the provided setup.py file:

::

    python setup.py install

Make sure you first install the required packages:

::

    pip install -r requirements.txts

FASTR tools
^^^^^^^^^^^

When running the FASTR package with a version lower than 1.3.0, you need
to manually add the PREDICT fastr\_tools path to the FASTR tools path.
Go the your FASTR config file (default: ~/.fastr/config.py) and add the
fastr\_tools path analogue to the description in the
PREDICT/fastrconfig/PREDICT\_config.py file:

::

    packagedir = site.getsitepackages()[0]
    tools_path = [os.path.join(packagedir, 'PREDICT', 'fastr_tools')] + tools_path

When using FASTR >1.3.0, the PREDICT config file will be automatically
created for you in the default: ~/.fastr/config.d folder.

Note that the Python site package does not work properly in virtual
environments. You must then manually locate the packagedir.

Preprocessing
~~~~~~~~~~~~~

From version 1.0.2 and on, preprocessing has been removed from PREDICT.
It is now available as a separate tool in the `WORC
package <https://github.com/MStarmans91/WORC>`__, as it's also a
separate step in the radiomics workflow. We do advice to use the
preprocessing function and thus also WORC.

3rd-party packages used in PREDICT:
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

We mainly rely on the following packages:

-  SimpleITK (Image loading and preprocessing)
-  numpy (Feature computation)
-  sklearn, scipy (Classification)
-  FASTR (Fast and parallel workflow execution)
-  pandas (Storage)
-  PyRadiomics

See also the `requirements file <requirements.txt>`__.

License
~~~~~~~

This package is covered by the open source `APACHE 2.0
License <APACHE-LICENSE-2.0>`__.

Contact
~~~~~~~

We are happy to help you with any questions: please send us a message or
create an issue on Github.
