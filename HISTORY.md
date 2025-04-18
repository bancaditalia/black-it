Release History
===============

0.3.2 (2025-04-05)
------------------

* Fix PytestRemovedIn8Warning ("support for nose tests is deprecated and will be removed in a future release") by @muxator in https://github.com/bancaditalia/black-it/pull/84
* tests: improve the signal/noise ratio of the test logs by 63% by @muxator in https://github.com/bancaditalia/black-it/pull/85
* tests: centralize --doctest-modules option in pytest.ini by @muxator in https://github.com/bancaditalia/black-it/pull/86
* dev: use poetry's post 1.2 format for specifying dev dependencies by @muxator in https://github.com/bancaditalia/black-it/pull/88
* ci: extract poetry and tox versions from the lockfile by @muxator in https://github.com/bancaditalia/black-it/pull/92
* plot bugfix: samplers pickle to scheduler_pickled by @aldoglielmo in https://github.com/bancaditalia/black-it/pull/93
* checkpointing: migrate from tables to h5py by @aldoglielmo in https://github.com/bancaditalia/black-it/pull/94
* ci: remove MacOS-specific trick that was needed for tables by @muxator in https://github.com/bancaditalia/black-it/pull/95
* tox: use commands from the Makefile where possible, without duplicating them by @muxator in https://github.com/bancaditalia/black-it/pull/96
* chore: update pandas dependency to version 2.2.3 by @aldoglielmo in https://github.com/bancaditalia/black-it/pull/98

**Full Changelog**: https://github.com/bancaditalia/black-it/compare/v0.3.1...v0.3.2

0.3.1 (2024-09-11)
------------------
* tests: add missing "311" env in tox.ini by @muxator in https://github.com/bancaditalia/black-it/pull/81
* RFC: support python 3.12 (but drop 3.8) by @muxator in https://github.com/bancaditalia/black-it/pull/82

0.3.0 (2023-10-19)
------------------
* losses: add likelihood loss and test, improve base loss by @aldoglielmo in https://github.com/bancaditalia/black-it/pull/38
* chore: poetry.lock update by @marcofavoritobi in https://github.com/bancaditalia/black-it/pull/40
* Fix spelling errors in `HISTORY.md` returned by `scripts/spell-check.sh -r` by @marcofavoritobi in https://github.com/bancaditalia/black-it/pull/44
* chore: update PR request template by @marcofavoritobi in https://github.com/bancaditalia/black-it/pull/42
* `tox.ini` refactoring by @marcofavoritobi in https://github.com/bancaditalia/black-it/pull/41
* Fix tox python env id by @marcofavoritobi in https://github.com/bancaditalia/black-it/pull/51
* Upgrade dependencies by @marcofavoritobi in https://github.com/bancaditalia/black-it/pull/46
* Update copyright notice for year 2023 by @marcofavoritobi in https://github.com/bancaditalia/black-it/pull/43
* Enable CI on `macos-latest` by @marcofavoritobi in https://github.com/bancaditalia/black-it/pull/47
* Add `BaseSeedable` class. by @marcofavorito in https://github.com/bancaditalia/black-it/pull/39
* Enable CI on `windows-latest` by @marcofavoritobi in https://github.com/bancaditalia/black-it/pull/48
* Introduce `BaseScheduler` abstraction by @marcofavoritobi in https://github.com/bancaditalia/black-it/pull/52
* Chore/upgrade ipywidgets version by @marcofavoritobi in https://github.com/bancaditalia/black-it/pull/54
* test: remove hypothesis deadline in test_seedable tests by @marcofavoritobi in https://github.com/bancaditalia/black-it/pull/55
* Fix mypy tox task by @marcofavoritobi in https://github.com/bancaditalia/black-it/pull/56
* Add `RLScheduler` by @marcofavoritobi in https://github.com/bancaditalia/black-it/pull/53
* Upgrade dependencies and minor updates by @marcofavoritobi in https://github.com/bancaditalia/black-it/pull/59
* build: upgrade dev dependency versions by @marcofavoritobi in https://github.com/bancaditalia/black-it/pull/60
* Skip plotting tests on Windows due to flakiness by @marcofavoritobi in https://github.com/bancaditalia/black-it/pull/63
* test: skip TestPlotConvergence test on Windows by @marcofavoritobi in https://github.com/bancaditalia/black-it/pull/64
* ci: fix flaky tox command on Windows platform (fix #65) by @marcofavoritobi in https://github.com/bancaditalia/black-it/pull/66
* dev: add missing .PHONY declarations by @marcofavoritobi in https://github.com/bancaditalia/black-it/pull/67
* gp sampler: GPy -> scikit-learn by @aldoglielmo in https://github.com/bancaditalia/black-it/pull/69
* Minor fix to tests by @marcofavoritobi in https://github.com/bancaditalia/black-it/pull/68
* Migrate to Ruff by @marcofavoritobi in https://github.com/bancaditalia/black-it/pull/62
* lint: move 'include' black and mypy config into `setup.cfg` by @marcofavoritobi in https://github.com/bancaditalia/black-it/pull/71
* Mypy fix ignores by @marcofavoritobi in https://github.com/bancaditalia/black-it/pull/70
* fix: typos in `.PHONY` directives in `Makefile` by @marcofavoritobi in https://github.com/bancaditalia/black-it/pull/74
* Update `poetry.lock` and `pyproject.toml`, add CI check on `poetry.lock` by @marcofavoritobi in https://github.com/bancaditalia/black-it/pull/72
* build: make the package to support Python 3.11 by @marcofavoritobi in https://github.com/bancaditalia/black-it/pull/73

0.2.1 (2023-02-06)
------------------
* samplers: add MLSurrogateSampler base class by @aldoglielmo in https://github.com/bancaditalia/black-it/pull/33
* xgboost: add clipping of loss values to the float32 limits by @aldoglielmo in https://github.com/bancaditalia/black-it/pull/35
* Update poetry.lock by @marcofavorito in https://github.com/bancaditalia/black-it/pull/34

0.2.0 (2022-12-21)
------------------
* Add joss references by @aldoglielmo in https://github.com/bancaditalia/black-it/pull/23
* add filters before loss function computation by @aldoglielmo in https://github.com/bancaditalia/black-it/pull/24
* calibrator: add sim_length param to Calibrator constructor by @aldoglielmo in https://github.com/bancaditalia/black-it/pull/25
* fix: add coordinate_filters to concrete loss classes by @marcofavoritobi in https://github.com/bancaditalia/black-it/pull/26
* Use tables to append to series_samp array by @aldoglielmo in https://github.com/bancaditalia/black-it/pull/27
* Improve best batch and random forest classes by @aldoglielmo in https://github.com/bancaditalia/black-it/pull/29
* Add CORS sampler by @aldoglielmo in https://github.com/bancaditalia/black-it/pull/28
* Add xgboost sampler by @aldoglielmo in https://github.com/bancaditalia/black-it/pull/30
* Refactoring and improvement of msm covariance matrices by @aldoglielmo in https://github.com/bancaditalia/black-it/pull/31

0.1.2 (2022-11-02)
------------------
* Dependencies: regenerate poetry.lock by @muxator in https://github.com/bancaditalia/black-it/pull/9
* Simplification of CIs by @aldoglielmo in https://github.com/bancaditalia/black-it/pull/13
* CIs: add upload of codecoverage to Codecov by @aldoglielmo in https://github.com/bancaditalia/black-it/pull/15
* Readme: add Codecov badge by @aldoglielmo in https://github.com/bancaditalia/black-it/pull/16
* Improved documentation by @aldoglielmo in https://github.com/bancaditalia/black-it/pull/10
* Update poetry.lock by @marcofavoritobi in https://github.com/bancaditalia/black-it/pull/18
* Particle swarm sampler by @marcofavoritobi in https://github.com/bancaditalia/black-it/pull/17

0.1.1 (2022-06-21)
------------------
* First release