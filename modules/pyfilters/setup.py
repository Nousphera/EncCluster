# Modified 2026-10-15: declare Python and NumPy requirements.
from setuptools import setup, find_packages

setup(
	name="pyfilters",
	version="1.0.0",
	description="Python bindings for C implementation of XOR filters",
	author="Vasileios Tsouvalas",
	author_email="v.tsouvalas@tue.nl",
	license="Apache 2.0",
	python_requires=">=3.11",
	packages=find_packages(),
    license_files=["pyfilters/lib/LICENSE"],
    package_data={"pyfilters":["lib/*.h", "lib/LICENSE"]},
	ext_package="pyfilters",
	install_requires=["cffi>=1.17", "numpy>=1.26,<3"],
	setup_requires=["cffi"],
	cffi_modules=["pyfilters/ffibuild.py:ffi"],
)
