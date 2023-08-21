from setuptools import setup, find_packages
from os import path

here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(here, 'README.md')) as f:
    long_description = f.read()

setup(
    # $ pip install mushroom
    name='mushroom',
    version='0.0.2',
    description='A Python library for clustering and analysis of multi-modal 3D serial section experiments.',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/dinglab/mushroom',
    author='Erik Storrs',
    author_email='estorrs@wustl.edu',
    classifiers=[
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
    ],
    keywords='multiplex imaging codex neighborhood analysis image segmentation visualization mibi codex phenocycler mihc hyperion',
    packages=find_packages(),
    python_requires='>=3.8',
    install_requires=[
        'scanpy',
        'seaborn',
        'tifffile',
        'ome-types',
        'imagecodecs>=2022.7.27',
        'scikit-image',
        'scikit-learn',
        'torch',
        'torchvision',
        'pytorch-lightning',
        'vit-pytorch',
        'einops',
        'wandb',
        'napari',
    ],
    include_package_data=True,
    entry_points={
        'console_scripts': [
 #           'mushroom=mushroom.mushroom:main',
        ],
    },
)
