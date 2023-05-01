set -e
rm -rf docs/generated/

read -p "(Recommended) Do you want to lint? (yes/no) " yn

case $yn in
	yes ) isort . ; black . ;;
	* ) echo Not linting.
esac

coverage run -m unittest
ruff .
sphinx-build -W docs docs/_build/html
