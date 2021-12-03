
from setuptools import find_packages
from setuptools import setup

from isl import __version__

with open('./README.md') as f:
    LONG_DESCRIPTION = f.read()

setup(
    name='isl',
    version=__version__,
    description='ISL: Inverse Supervised Learning',
    long_description=LONG_DESCRIPTION,
    download_url='https://github.com/HenryLee97/isl',
    author='Changu Kang, Seokhyun Lee',
    python_version='>=3.7',
    packages=find_packages(include=('isl', 'isl.*')),
    include_package_data=True,
    setup_requires=[],
    install_requires=[
        'torch>=1.10.0',
        'torchvision>=0.11.0',
    ],
    dependency_links=[],
)
