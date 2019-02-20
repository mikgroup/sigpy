import sys
from setuptools import setup, find_packages

if sys.version_info < (3, 5):
    sys.exit('Sorry, Python < 3.5 is not supported')

REQUIRED_PACKAGES = ['numpy', 'pywavelets', 'numba', 'tqdm']

with open("README.rst", "r") as f:
    long_description = f.read()

setup(name='sigpy',
      version='0.1.0',
      description='Python package for signal reconstruction.',
      long_description=long_description,
      long_description_content_type="text/markdown",
      url='http://github.com/mikgroup/sigpy',
      author='Frank Ong',
      author_email='frankong@berkeley.edu',
      license='BSD',
      packages=find_packages(),
      install_requires=REQUIRED_PACKAGES,
      scripts=['bin/image_plot',
               'bin/line_plot',
               'bin/scatter_plot'],
      classifiers=[
          "Programming Language :: Python :: 3",
          "License :: OSI Approved :: BSD License",
          "Operating System :: OS Independent",
      ]
)
