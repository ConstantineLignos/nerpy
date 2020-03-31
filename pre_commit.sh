#! /bin/bash
set -euxo pipefail
files='nerpy/ scripts/ tests/ *.py'
isort -rc $files
black $files
flake8 $files
mypy $files
pytest --cov-report term-missing --cov=nerpy tests/
