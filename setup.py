from gettext import install
from setuptools import setup, find_packages

setup(
    name='amelia_datatools',
    packages=find_packages(['./tools/*'], exclude=['test*']),
    version='1.0',
    description='Tools for Amelia dataset analisis',
    install_requires=[
        'numpy==1.21.2,<2',
        'matplotlib==3.7.1',
        'pandas',
        'tqdm',
        "easydict",
        "pyproj==3.6.1",
        'scipy==1.9.1',
        "seaborn",
    ]
)
