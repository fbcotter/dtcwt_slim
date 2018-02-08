import os
import re

from setuptools import setup, find_packages


# Utility function to read the README file.
# Used for the long_description.  It's nice, because now 1) we have a top level
# README file and 2) it's easier to type in the README file than to put a raw
# string in below ...
def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


# Read metadata from version file
metadata_file = open(os.path.join(os.path.dirname(__file__), 'dtcwt_slim', '_version.py')).read()
metadata = dict(re.findall("__([a-z]+)__ = '([^']+)'", metadata_file))

setup(
    name='dtcwt_slim',
    version=metadata['version'],
    author="Fergal Cotter",
    author_email="fbc23@cam.ac.uk",
    description=("A slim port of the DTCWT toolbox to run on tensorflow"),
    license="Free To Use But Restricted",
    keywords="numpy, wavelet, complex wavelet, DT-CWT",
    url="https://github.com/fbcotter/dtcwt_slim",
    packages=find_packages(exclude=["*.tests", "*.tests.*", "tests.*", "tests"]),
    long_description=read('README.rst'),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "License :: Free To Use But Restricted",
        "Programming Language :: Python :: 3",
    ],
    include_package_data=True,
    install_requires=['numpy', 'six', ],

    extras_require={
        'docs': ['sphinx', 'docutils', 'matplotlib', 'ipython', ],
        'opencl': ['pyopencl', ],
    },

    tests_require=['coverage', 'py3nvml', 'dtcwt'],
)

# vim:sw=4:sts=4:et
