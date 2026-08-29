.PHONY: install test reproduce app clean
install:
	python -m pip install -e .[dev]
test:
	pytest -q
reproduce: test
	python -m tasksemantic.experiments
app:
	python app.py
clean:
	rm -rf .pytest_cache build dist *.egg-info src/*.egg-info src/tasksemantic/__pycache__ tests/__pycache__
