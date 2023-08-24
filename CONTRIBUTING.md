# Extending Black-it

Contributions to the library are welcome, and they are greatly appreciated! Every little bit helps, and credit will always be given.

There are various ways to contribute:

- If you need support, want to report a bug or ask for features, you can check the [Issues page](https://github.com/bancaditalia/black-it/issues) and raise an issue, if applicable.

- If you would like to contribute a bug fix of feature then [submit a Pull request](https://github.com/bancaditalia/black-it/pulls).

For other kinds of feedback, you can contact one of the
[authors](https://github.com/bancaditalia/black-it/blob/main/AUTHORS.md) by email.


## A few simple rules

- All pull requests should be opened against the `develop` branch. Do **not** open a Pull Request against `main`.

- Before working on a feature, reach out to one of the core developers or discuss the feature in an issue. The library caters a diverse audience and new features require upfront coordination.

- Include unit tests for 100% coverage when you contribute new features, as they help to a) prove that your code works correctly, and b) guard against future breaking changes to lower the maintenance cost.

- Bug fixes also generally require unit tests, because the presence of bugs usually indicates insufficient test coverage.

- Whenever possible, keep API compatibility in mind when you change code in the `black_it` library. Reviewers of your pull request will comment on any API compatibility issues.

- All files must include a license header.

- Before committing and opening a PR, run all tests locally. This saves CI hours and ensures you only commit clean code.


## Contributing code

If you have improvements, send us your pull requests!

A team member will be assigned to review your pull requests. All tests are run as part of CI as well as various other checks (linters, static type checkers, security checkers, etc). If there are any problems, feedback is provided via GitHub. Once the pull request is approved and passes continuous integration checks, you or a team member can merge it.

If you want to contribute, start working through the codebase, navigate to the GitHub "issues" tab and start looking through interesting issues. If you decide to start on an issue, leave a comment so that other people know that you're working on it. If you want to help out, but not alone, use the issue comment thread to coordinate.


## Development setup

Set up your development environment by following these steps:

- Install [`Poetry`](https://python-poetry.org/), either by running run `pip install poetry` or as indicated [here](https://python-poetry.org/docs/#installation).

- Get the latest version of the code by running

```
git clone https://github.com/bancaditalia/black-it.git
cd black-it
```

- Setup a Poetry environment by running

```
poetry shell
poetry install
```

## Further commands needed during development

We have various commands which are helpful during development.

- For linting and static analysis use:
```
make lint
make static
make pylint
make safety
make bandit
```

- To apply [`black`](https://black.readthedocs.io/en/stable/) code formatter:
```
make black
```

- whereas, to only check compliance:
```
make black-check
```

- To run tests: `make test`.

- For testing `black_it.{SUBMODULE}` with `tests/test_{TESTMODULE}` use:
```
make test-sub dir={SUBMODULE} tdir={TESTMODULE}
```

e.g.
```
make test-sub tdir=losses dir=loss_functions
```
