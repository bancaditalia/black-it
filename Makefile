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

.PHONY: help
help:
	@python -c "$$PRINT_HELP_PYSCRIPT" < $(MAKEFILE_LIST)

.PHONY: clean
clean: clean-build clean-pyc clean-test clean-docs ## remove all build, test, coverage and Python artifacts

.PHONY: clean-build
clean-build: ## remove build artifacts
	rm -fr build/
	rm -fr dist/
	rm -fr .eggs/
	find . -name '*.egg-info' -exec rm -fr {} +
	find . -name '*.egg' -exec rm -f {} +

.PHONY: clean-pyc
clean-pyc: ## remove Python file artifacts
	find . -name '*.pyc' -exec rm -f {} +
	find . -name '*.pyo' -exec rm -f {} +
	find . -name '*~' -exec rm -f {} +
	find . -name '__pycache__' -exec rm -fr {} +

.PHONY: clean-docs
clean-docs:  ## remove MkDocs products.
	mkdocs build --clean
	rm -fr site/


.PHONY: clean-test
clean-test: ## remove test and coverage artifacts
	rm -fr .tox/
	rm -f .coverage
	rm -fr htmlcov/
	rm -fr .pytest_cache
	rm -fr .mypy_cache
	rm -fr coverage.xml

.PHONY: lint-all
lint-all: black ruff static bandit safety vulture darglint ## run all linters

.PHONY: lint-all-files
lint-all-files: black-files ruff-files static-files bandit-files vulture-files darglint-files ## run all linters for specific files (specified with files="file1 file2 somedir ...")

.PHONY: poetry-lock-check
poetry-lock-check: ## check if poetry.lock is consistent with pyproject.toml
	poetry check --lock

.PHONY: static
static: ## static type checking with mypy
	mypy

.PHONY: static-files
static-files: ## static type checking with mypy for specific files (specified with files="file1 file2 somedir ...")
	$(call check_defined, files)
	mypy $(files)

.PHONY: black
black: ## apply black formatting
	black .

.PHONY: black-files
black-files: ## apply black formatting for specific files (specified with files="file1 file2 somedir ...")
	$(call check_defined, files)
	black $(files)

.PHONY: black-check
black-check: ## check black formatting
	black --check --verbose .

.PHONY: black-check-files
black-check-files: ## check black formatting for specific files (specified with files="file1 file2 somedir ...")
	$(call check_defined, files)
	black --check --verbose $(files)

.PHONY: ruff
ruff: ## run ruff linter
	ruff check --fix --show-fixes .

.PHONY: ruff-files
ruff-files: ## run ruff linter for specific files (specified with files="file1 file2 somedir ...")
	$(call check_defined, files)
	ruff check --fix --show-fixes $(files)

.PHONY: ruff-check
ruff-check: ## check ruff linter rules
	ruff check .

.PHONY: ruff-check-files
ruff-check-files: ## check ruff linter rules for specific files (specified with files="file1 file2 somedir ...")
	$(call check_defined, files)
	ruff check $(files)

.PHONY: bandit
bandit: ## run bandit
	bandit --configfile .bandit.yaml --recursive black_it tests scripts examples

.PHONY: bandit-files
bandit-files: ## run bandit for specific files (specified with files="file1 file2 somedir ...")
	$(call check_defined, files)
	bandit $(files)

.PHONY: safety
safety: ## run safety
	safety check

.PHONY: vulture
vulture: ## run vulture
	vulture black_it scripts/whitelists/package_whitelist.py
	vulture examples scripts/whitelists/examples_whitelist.py
	vulture tests scripts/whitelists/tests_whitelist.py

.PHONY: vulture-files
vulture-files: ## run vulture for specific files (specified with files="file1 file2 somedir ...")
	$(call check_defined, files)
	vulture $(files) scripts/whitelists/package_whitelist.py scripts/whitelists/examples_whitelist.py scripts/whitelists/tests_whitelist.py

.PHONY: darglint
darglint: ## run darglint
	darglint black_it

.PHONY: darglint-files
darglint-files: ## run darglint for specific files (specified with files="file1 file2 somedir ...")
	$(call check_defined, files)
	darglint $(files)

.PHONY: test
test: ## run tests quickly with the default Python
	pytest                              \
		tests/                          \
		--doctest-modules black_it      \
		--cov=black_it                  \
		--cov-report=xml                \
		--cov-report=html               \
		--cov-report=term               \
		-m 'not e2e'

.PHONY: test-e2e
test-e2e:
	pytest tests                        \
		--cov=black_it                  \
		--cov-report=xml                \
		--cov-report=html               \
		--cov-report=term               \
		-m 'e2e'

.PHONY: test-nb
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


.PHONY: test-all
test-all: ## run tests on every Python version with tox
	tox

.PHONY: coverage
coverage: ## check code coverage quickly with the default Python, omitting example models (not really part of black-it)
	coverage run --source black_it --omit="*/examples/models*,*/black_it/plot*" -m pytest
	coverage report -m
	coverage html
	$(BROWSER) htmlcov/index.html

.PHONY: docs
docs: ## generate MkDocs HTML documentation, including API docs
	mkdocs build --clean
	$(BROWSER) site/index.html

.PHONY: servedocs
servedocs: docs ## compile the docs watching for changes
	mkdocs build --clean
	python -c 'print("###### Starting local server. Press Control+C to stop server ######")'
	mkdocs serve

.PHONY: release
release: dist ## package and upload a release
	twine upload dist/*

.PHONY: dist
dist: clean ## builds source and wheel package
	poetry build
	ls -l dist

.PHONY: install
install: clean ## install the package to the active Python's site-packages
	poetry install

.PHONY: develop
develop: clean ## install the package in development mode
	echo "Not supported by Poetry yet!"

check_defined = \
    $(if -nz $(value $1),, \
      $(error Undefined $1$(if $2, ($2))))
