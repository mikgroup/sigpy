import os
import sys
from setuptools import setup

if sys.version_info < (3, 0):
    sys.exit('Sorry, Python < 3.0 is not supported')

REQUIRED_PACKAGES = ['numpy', 'pywavelets', 'numba']

with open("README.md", "r") as f:
    long_description = f.read()

setup(name='sigpy',
      version='0.0.1',
      description='Python package for signal reconstruction.',
      long_description=long_description,
      long_description_content_type="text/markdown",
      url='http://github.com/mikgroup/sigpy',
      author='Frank Ong',
      author_email='frankong@berkeley.edu',
      license='BSD',
      packages=['sigpy'],
      install_requires=REQUIRED_PACKAGES,
      scripts=['bin/sigpy_plot'],
      classifiers=(
          "Programming Language :: Python :: 3",
          "License :: OSI Approved :: BSD License",
          "Operating System :: OS Independent",
      )
)
