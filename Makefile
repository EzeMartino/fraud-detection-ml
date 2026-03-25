install:
	pip install -r requirements.dev.txt

train:
	python -m src.models.tune_random_forest
	python -m src.models.train_random_forest

test:
	pytest

serve:
	uvicorn src.api.main:app --reload

lint:
	ruff check .

profile:
	python -m src.data.load_and_profile