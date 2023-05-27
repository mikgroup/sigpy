import sys

from setuptools import find_packages, setup

from sigpy.version import __version__  # noqa

if sys.version_info < (3, 6):
    sys.exit("Sorry, Python < 3.6 is not supported")

REQUIRED_PACKAGES = ["numpy", "pywavelets", "numba", "scipy", "tqdm"]

with open("README.rst", "r") as f:
    long_description = f.read()

setup(
    name="sigpy",
    version=__version__,
    description="Python package for signal reconstruction.",
    long_description=long_description,
    long_description_content_type="text/x-rst",
    url="http://github.com/mikgroup/sigpy",
    author="Frank Ong",
    author_email="frankong@berkeley.edu",
    license="BSD",
    packages=find_packages(),
    install_requires=REQUIRED_PACKAGES,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: BSD License",
        "Operating System :: OS Independent",
    ],
)
