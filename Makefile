# Canonical verify loop — `make check` is the self-grading gate for this repo.
# Deps come from the dev dependency-group (ruff, mypy, pytest, pillow), which
# `uv run` includes by default.
.PHONY: check
check:
	uv run ruff check .
	uv run mypy saccade
	uv run python -m pytest -q
