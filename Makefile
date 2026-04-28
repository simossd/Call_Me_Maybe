NAME = src

install:
        @uv sync

run:
        @uv run python -m $(NAME)

clean:
        @rm -rf */__pycache__ */.mypy_cache .mypy_cache __pycache__

.PHONY:	install run clean