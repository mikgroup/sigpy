set -e
rm -rf docs/generated/
black .
isort .
ruff check .
coverage run -m unittest
sphinx-build -W docs docs/_build/html
