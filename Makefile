all: dev build_ext

install:
	python3 -m pip install .

dev:
	python3 -m pip install -e .

build_ext:
	python3 setup.py build_ext --build-lib=.
