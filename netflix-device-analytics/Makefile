.PHONY: setup test lint run compose-up compose-down

setup:
	python -m pip install -r requirements.txt

test:
	pytest

lint:
	python -m compileall .

run:
	streamlit run dashboard/app.py --server.port $${STREAMLIT_SERVER_PORT:-8502}

compose-up:
	docker compose up --build

compose-down:
	docker compose down -v

