#!/usr/bin/env bash
set -e

# The notebook-facing helpers; the notebook itself is checked by running it.
uv run pylint case_explorer.py case_timeline.py
uv run mypy case_explorer.py case_timeline.py
