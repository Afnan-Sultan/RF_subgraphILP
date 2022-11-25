from distutils.core import setup

import numpy
from Cython.Build import cythonize

setup(
    name="BiasedCriterion",
    ext_modules=cythonize(
        "biased_criterion.pyx", compiler_directives={"language_level": "3"}
    ),
    include_dirs=[numpy.get_include()],
)
