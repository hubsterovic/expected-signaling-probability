.PHONY: test lint figures

test:
	uv run pytest -q

lint:
	uv run ruff check .

figures:
	MPLBACKEND=Agg uv run python -m expected_signaling_probability.experiments.make_figures
