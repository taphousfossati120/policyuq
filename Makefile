.PHONY: help install dev lint type test docker clean

help:
	@echo "install  install the package"
	@echo "dev      install with dev extras"
	@echo "lint     ruff + black + isort checks"
	@echo "type     mypy --strict"
	@echo "test     pytest"
	@echo "docker   build the container image"
	@echo "clean    remove caches and build artifacts"

install:
	pip install -e .

dev:
	pip install -e ".[dev,text]"

lint:
	ruff check .
	black --check .
	isort --check-only .

type:
	mypy --strict policyuq_vlm

test:
	pytest

docker:
	docker build -t policyuq-vlm .

clean:
	find . -name '__pycache__' -prune -exec rm -rf {} +
	find . -name '*.egg-info' -prune -exec rm -rf {} +
	rm -rf .mypy_cache .pytest_cache .ruff_cache runs
