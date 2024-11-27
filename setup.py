from setuptools import setup, find_packages
from pathlib import Path

# Project metadata
project_name = 'CarObjectDetection'
version = '0.1.0'
url = 'https://github.com/mauriceoboya/car_object_detection'
license_type = 'MIT'
author = 'Maurice Oboya'
description = 'A project to detect cars in images'
long_description = Path('README.md').read_text(encoding='utf-8')
long_description_content_type = 'text/markdown'

setup(
    name=project_name,
    version=version,
    description=description,
    author=author,
    license=license_type,
    long_description=long_description,
    long_description_content_type=long_description_content_type,
    package_dir={'': 'src'},  
    packages=find_packages(where='src'), 
    install_requires=[
        'numpy',
        'opencv-python',
        'pyyaml',
        'scikit-learn',
        'torchvision',
        'tqdm',
        'matplotlib',
        'pandas',
        'seaborn',
        'pillow',
    ],
    classifiers=[
        'Programming Language :: Python :: 3.7',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.7',
    project_urls={
        'Source Code': url,
        'Documentation': url,
    },
    keywords='car object detection machine learning',
)
