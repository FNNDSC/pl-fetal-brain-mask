from os import path
from setuptools import setup

with open(path.join(path.dirname(path.abspath(__file__)), 'README.md')) as f:
    readme = f.read()

setup(
    name             = 'fetal_brain_mask',
    version          = '1.1.0',
    description      = 'Automatic masking of fetal brain images using deep learning',
    long_description = readme,
    author           = 'Alejandro Valdes',
    author_email     = 'dev@babyMRI.org',
    url              = 'https://github.com/FNNDSC/pl-fetal-brain-mask',
    packages         = ['fetal_brain_mask'],
    license          = 'MIT',
    zip_safe         = False,
    python_requires  = '>=3.6',
    package_data     = {
        'fetal_brain_mask': ['data/*']
    },
    entry_points     = {
        'console_scripts': [
            'fetal_brain_mask = fetal_brain_mask.__main__:main'
            ]
        }
)
