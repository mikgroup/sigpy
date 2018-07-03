sigpy
=====
This is a Python package for signal reconstruction.

Requirements
------------
This package requires python3, numpy, scipy, pywavelets, and numba. For optional gpu support, the package requires cupy, and for optional parallel programming support, the package requires mpi4py.
The easiest way to setup for this package is to install python3 through Anaconda.

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
	
To get full performance on cpu, you can optionally install the intel based python. I recommend creating an environment:

	conda create -c intel -n idp intelpython3_full
	source activate idp

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
