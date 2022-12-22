from setuptools import setup

import numpy as np
from pathlib import Path
from transonic.dist import init_transonic_extensions, ParallelBuildExt, init_pythran_extensions
from transonic.backends import make_backend_files


def create_transonic_extensions():
    # Prepare each file that should be pythranized
    here = Path(__file__).parent.absolute()
    files_to_pythranize = ['src/lmc/misp_lmc.py']
    make_backend_files( [here / path for path in files_to_pythranize] )
    # Create extension for every pythranizable file in src/lmc
    transonic_extensions = init_transonic_extensions("src/lmc", backend="pythran", compile_args=("-O2", "-march=native", "-DUSE_XSIMD"), include_dirs=np.get_include(), inplace = 1)
#    transonic_extensions = init_pythran_extensions("src/lmc", compile_args=("-O2", "-march=native", "-DUSE_XSIMD"), include_dirs=np.get_include())
    return transonic_extensions


setup(ext_modules = create_transonic_extensions(), cmdclass={"build_ext": ParallelBuildExt})
#setup()
