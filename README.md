sigpy
=====
This is a Python package for signal reconstruction, with GPU support using cupy.

Requirements
------------
This package requires python3, numpy, scipy, pywavelets, and numba. 

For optional gpu support, the package requires cupy.

For optional distributed programming support, the package requires mpi4py.

Installation through Anaconda
-----------------------------

Install Anaconda:

Download Anaconda with Python 3 from this website: https://www.continuum.io/downloads
and follow their installation instruction.

Install required packages:

	conda install pywavelets numba

To enable gpu support, you can optionally install:

	conda install cupy
	
To enable mpi support, you can optionally install:

	conda install mpi4py
	
Finally to link the package, go to the main directory with the setup.py script and run:

    python setup.py develop

This allows importing the library at any location.

Installation through pip
------------------------

Install packages:

	pip install -r requirements.txt
	
To enable gpu support, you can optionally install:

	pip install cupy
	
To enable mpi support, you can optionally install:

	pip install mpi4py
	
Finally to link the package, go to the main directory with the setup.py script and run:

    python setup.py develop

This allows importing the library at any location.
