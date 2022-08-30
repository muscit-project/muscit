all: dev build_ext

install:
	python -m pip install .

dev:
	python -m pip install -e .

build_ext:
	python setup.py build_ext --build-lib=.
