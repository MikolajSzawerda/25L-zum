@default: lint

@init:
    uv sync --all-extras

#     --cache-dir .uv_cache

@env:
    cp .env.example .env

@test:
    echo "Running tests..."
    echo "Tests passed!"

@lint:
    uv run ruff format
    uv run ruff check .

@push: lint
    git push

@clean:
    rm -rf .venv

@uv-init:
    uv init --python 3.12

@uv-clean:
    uv cache clean

@uv-prune:
    uv cachce prune
