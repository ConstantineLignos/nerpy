#!/usr/bin/env bash
set -euxo pipefail

files=(nerpy/ tests/ setup.py)
black --check "${files[@]}"
flake8 "${files[@]}"
mypy "${files[@]}"
pytest tests/
