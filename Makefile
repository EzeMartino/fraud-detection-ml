install:
	pip install -r requirements.txt

test:
	pytest

lint:
	ruff check .

profile:
	python -m src.data.load_and_profile