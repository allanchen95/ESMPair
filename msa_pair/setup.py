from setuptools import find_packages
from setuptools import setup

setup(
    name='msa_pair',
    packages=find_packages(),
    install_requires=[
        'pandas',
        'scipy',
        'requests',
        'biopython',
    ],
)