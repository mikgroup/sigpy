coverage run -m unittest
flake8
sphinx-build -W doc doc/_build/html
