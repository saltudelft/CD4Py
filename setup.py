# License: GNU General Public License v3.0

from setuptools import setup
from os import path
from cd4py import __version__

here = path.abspath(path.dirname(__file__))

with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='cd4py',
    version=__version__,
    description='CD4Py: Code De-Duplication for Python',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/saltudelft/CD4Py',
    author='Amir M. Mir (TU Delft)',
    author_email='mir-am@hotmail.com',
    classifiers=[
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Environment :: Console',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Operating System :: Unix',
        'Operating System :: MacOS',
        'Operating System :: POSIX',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Software Development :: Libraries :: Python Modules'
    ],
    keywords='python source code de-duplication detection cd4py',
    packages=['cd4py'],
    python_requries='>=3.5',
    install_requires=['dpu-utils', 'tqdm', 'joblib', 'pandas', 'scikit-learn', 'numpy', 'annoy'],
    entry_points={
        'console_scripts': [
            'cd4py = cd4py.__main__:main',
        ],
    }
)
