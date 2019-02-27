from setuptools import setup, find_packages
import sys
from os import path
from io import open

if sys.version_info < (3,5):
    sys.exit("Python < 3.5")

here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(here, 'README.md'), encoding='utf-8') as f:
        long_description = f.read()


setup(name='comm_bench',
      version='0.0.1',
      description="Collection of microbenchmarks for Chainer communicators",
      long_description_content_type='text/markdown',
      author='Uenishi Kota',
      author_email='kota@preferred.jp',

      long_description=long_description,
      packages=find_packages(exclude=['contrib', 'docs', 'tests']),

      install_requires=['mpi4py', 'chainer', 'chainercv', 'numpy', 'matplotlib','tqdm'],
      entry_points={'console_scripts':
                    ['comm_bench1=comm_bench.bench1:main',
                     'comm_bench2=comm_bench.bench2:main',]},
      python_requires='>=3.5',
      classifiers=[
                'Development Status :: 3 - Alpha',
                'Intended Audience :: Developers',
                'License :: OSI Approved :: MIT License',
                'Programming Language :: Python :: 3',
                'Programming Language :: Python :: 3.5',
                'Programming Language :: Python :: 3.6',
                'Programming Language :: Python :: 3.7',
            ],
      keywords='chainer chainermn nccl',
      )
