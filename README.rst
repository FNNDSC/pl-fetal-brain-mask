pl-fetal-brain-masking
================================

.. image:: https://img.shields.io/docker/v/fnndsc/pl-fetal-brain-masking?sort=semver
    :target: https://hub.docker.com/r/fnndsc/pl-fetal-brain-masking

.. image:: https://img.shields.io/github/license/fnndsc/pl-fetal-brain-masking
    :target: https://github.com/FNNDSC/pl-fetal-brain-masking/blob/master/LICENSE


.. contents:: Table of Contents


Abstract
--------

An app to ...


Description
-----------

``fetal_brain_masking`` is a ChRIS-based application that...


Usage
-----

.. code::

    python fetal_brain_masking.py
        [-h|--help]
        [--json] [--man] [--meta]
        [--savejson <DIR>]
        [-v|--verbosity <level>]
        [--version]
        <inputDir> <outputDir>


Arguments
~~~~~~~~~

.. code::

    [-h] [--help]
    If specified, show help message and exit.
    
    [--json]
    If specified, show json representation of app and exit.
    
    [--man]
    If specified, print (this) man page and exit.

    [--meta]
    If specified, print plugin meta data and exit.
    
    [--savejson <DIR>] 
    If specified, save json representation file to DIR and exit. 
    
    [-v <level>] [--verbosity <level>]
    Verbosity level for app. Not used currently.
    
    [--version]
    If specified, print version number and exit. 


Getting inline help is:

.. code:: bash

    docker run --rm fnndsc/pl-fetal-brain-masking fetal_brain_masking --man

Run
~~~

You need you need to specify input and output directories using the `-v` flag to `docker run`.


.. code:: bash

    docker run --rm -u $(id -u)                             \
        -v $(pwd)/in:/incoming -v $(pwd)/out:/outgoing      \
        fnndsc/pl-fetal-brain-masking fetal_brain_masking                        \
        /incoming /outgoing


Development
-----------

Build the Docker container:

.. code:: bash

    docker build -t local/pl-fetal-brain-masking .

Run unit tests:

.. code:: bash

    docker run --rm local/pl-fetal-brain-masking nosetests

Examples
--------

Put some examples here!


.. image:: https://raw.githubusercontent.com/FNNDSC/cookiecutter-chrisapp/master/doc/assets/badge/light.png
    :target: https://chrisstore.co
