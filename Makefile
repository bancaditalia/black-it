.PHONY: clean clean-test clean-pyc clean-build docs help
.DEFAULT_GOAL := help

define BROWSER_PYSCRIPT
import os, webbrowser, sys

try:
	from urllib import pathname2url
except:
	from urllib.request import pathname2url

webbrowser.open("file://" + pathname2url(os.path.abspath(sys.argv[1])))
endef
export BROWSER_PYSCRIPT

define PRINT_HELP_PYSCRIPT
import re, sys

for line in sys.stdin:
	match = re.match(r'^([0-9a-zA-Z_-]+):.*?## (.*)$$', line)
	if match:
		target, help = match.groups()
		print("%-20s %s" % (target, help))
endef
export PRINT_HELP_PYSCRIPT

BROWSER := python -c "$$BROWSER_PYSCRIPT"

help:
	@python -c "$$PRINT_HELP_PYSCRIPT" < $(MAKEFILE_LIST)

clean: clean-build clean-pyc clean-test clean-docs ## remove all build, test, coverage and Python artifacts

clean-build: ## remove build artifacts
	rm -fr build/
	rm -fr dist/
	rm -fr .eggs/
	find . -name '*.egg-info' -exec rm -fr {} +
	find . -name '*.egg' -exec rm -f {} +

clean-pyc: ## remove Python file artifacts
	find . -name '*.pyc' -exec rm -f {} +
	find . -name '*.pyo' -exec rm -f {} +
	find . -name '*~' -exec rm -f {} +
	find . -name '__pycache__' -exec rm -fr {} +

clean-docs:  ## remove MkDocs products.
	mkdocs build --clean
	rm -fr site/


clean-test: ## remove test and coverage artifacts
	rm -fr .tox/
	rm -f .coverage
	rm -fr htmlcov/
	rm -fr .pytest_cache
	rm -fr .mypy_cache
	rm -fr coverage.xml

lint-all: black isort flake8 static bandit safety vulture darglint pylint ## run all linters

lint-all-files: black-files isort-files flake8-files static-files bandit-files vulture-files darglint-files pylint-files ## run all linters for specific files (specified with files="file1 file2 somedir ...")

flake8: ## check style with flake8
	flake8 black_it tests scripts examples

flake8-files: ## check style with flake8 for specific files (specified with files="file1 file2 somedir ...")
	$(call check_defined, files)
	flake8 $(files)

static: ## static type checking with mypy
	mypy black_it tests scripts examples

static-files: ## static type checking with mypy for specific files (specified with files="file1 file2 somedir ...")
	$(call check_defined, files)
	mypy $(files)

isort: ## sort import statements with isort
	isort black_it tests scripts examples

isort-files: ## sort import statements with isort for specific files (specified with files="file1 file2 somedir ...")
	$(call check_defined, files)
	isort $(files)

isort-check: ## check import statements order with isort
	isort --check-only black_it tests scripts examples

isort-check-files: ## check import statements order with isort for specific files (specified with files="file1 file2 somedir ...")
	$(call check_defined, files)
	isort --check-only $(files)

black: ## apply black formatting
	black black_it tests scripts examples

black-files: ## apply black formatting for specific files (specified with files="file1 file2 somedir ...")
	$(call check_defined, files)
	black $(files)

black-check: ## check black formatting
	black --check --verbose black_it tests scripts examples

black-check-files: ## check black formatting for specific files (specified with files="file1 file2 somedir ...")
	$(call check_defined, files)
	black --check --verbose $(files)

bandit: ## run bandit
	bandit --configfile .bandit.yaml --recursive black_it tests scripts examples

bandit-files: ## run bandit for specific files (specified with files="file1 file2 somedir ...")
	$(call check_defined, files)
	bandit $(files)

safety: ## run safety
	safety check -i 44715 -i 44716 -i 44717 -i 47794

pylint: ## run pylint
	pylint black_it tests scripts examples

pylint-files: ## run pylint for specific files (specified with files="file1 file2 somedir ...")
	$(call check_defined, files)
	pylint $(files)

vulture: ## run vulture
	vulture black_it scripts/whitelists/package_whitelist.py
	vulture examples scripts/whitelists/examples_whitelist.py
	vulture tests scripts/whitelists/tests_whitelist.py

vulture-files: ## run vulture for specific files (specified with files="file1 file2 somedir ...")
	$(call check_defined, files)
	vulture $(files) scripts/whitelists/package_whitelist.py scripts/whitelists/examples_whitelist.py scripts/whitelists/tests_whitelist.py

darglint: ## run darglint
	darglint black_it

darglint-files: ## run darglint for specific files (specified with files="file1 file2 somedir ...")
	$(call check_defined, files)
	darglint $(files)

test: ## run tests quickly with the default Python
	pytest                              \
		tests/                          \
		--doctest-modules black_it      \
		--cov=black_it                  \
		--cov-report=xml                \
		--cov-report=html               \
		--cov-report=term               \
		-m 'not e2e'

test-e2e:
	pytest tests                        \
		--cov=black_it                  \
		--cov-report=xml                \
		--cov-report=html               \
		--cov-report=term               \
		-m 'e2e'

test-nb:
	pytest examples --nbmake --nbmake-timeout=300


# how to use:
#
#     make test-sub tdir=$TDIR dir=$DIR
#
# where:
# - TDIR is the path to the test module/directory (but without the leading "test_")
# - DIR is the *dotted* path to the module/subpackage whose code coverage needs to be reported.
#
# For example, to run the loss function tests (in tests/test_losses)
# and check the code coverage of the package black_it.loss_functions:
#
#     make test-sub tdir=losses dir=loss_functions
#
.PHONY: test-sub
test-sub:
	pytest -rfE tests/test_$(tdir) --cov=black_it.$(dir) --cov-report=html --cov-report=xml --cov-report=term-missing --cov-report=term  --cov-config=.coveragerc
	find . -name ".coverage*" -not -name ".coveragerc" -exec rm -fr "{}" \;


test-all: ## run tests on every Python version with tox
	tox

coverage: ## check code coverage quickly with the default Python, omitting example models (not really part of black-it)
	coverage run --source black_it --omit="*/examples/models*,*/black_it/plot*" -m pytest
	coverage report -m
	coverage html
	$(BROWSER) htmlcov/index.html

docs: ## generate MkDocs HTML documentation, including API docs
	mkdocs build --clean
	$(BROWSER) site/index.html

servedocs: docs ## compile the docs watching for changes
	mkdocs build --clean
	python -c 'print("###### Starting local server. Press Control+C to stop server ######")'
	mkdocs serve

release: dist ## package and upload a release
	twine upload dist/*

dist: clean ## builds source and wheel package
	poetry build
	ls -l dist

install: clean ## install the package to the active Python's site-packages
	poetry install

develop: clean ## install the package in development mode
	echo "Not supported by Poetry yet!"

check_defined = \
    $(if -nz $(value $1),, \
      $(error Undefined $1$(if $2, ($2))))
