Contribution Guide
------------------

Thank you for considering contributing to SigPy. 
To contribute to the source code, you can submit a `pull request <https://github.com/mikgroup/sigpy/pulls>`_.

To ensure your pull request can be merged quickly, you should check the following three items:

- `Coding Style`_
- `Unit Testing`_
- `Documentation`_

A simple way to check is to run ``run_tests.sh``. You will need to install::

$ pip install codecov flake8 sphinx sphinx_rtd_theme matplotlib

Any new features (new functions, Linops, Apps...), bug fixing, and improved documentation are welcome. 
We only ask you to avoid replicating existing features in NumPy, CuPy, and SigPy.
A general rule is that if a feature can already be implemented in one line, 
then it is probably not worth defining as a new function.

Coding Style
============

SigPy adopts the `Google code style <http://google.github.io/styleguide/pyguide.html>`_.
In particular, docstrings should use ``Args`` and ``Returns`` as described in the `Comments and Docstrings section <http://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings>`_.

You can use ``autopep8`` and ``flake8`` to check your code.

Unit Testing
============

You should write test cases for each function you commit. The unit tests are under the directory ``tests/``. 
The file hierarchy should follow the ``sigpy/`` directory, but each file should be prepended by ``test_``.

SigPy use the ``unittest`` package for testing. You can run tests by doing::

$ python -m unittest

Documentation
=============

Each new feature should be documented in the documentation. The documention is stored under the directory``docs/``.

You can build the docmentation in HTML format locally using Sphinx::

$ cd docs
$ make html
