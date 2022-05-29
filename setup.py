from distutils.core import setup

setup(
      name = 'covid',
      version = '0.1.0',
      url = 'https: //github.com/bnb32/covid_modeling',
      author = 'Brandon N. Benton',
      description = 'various utilities for modeling covid',
      packages = ['covid'],
      package_dir = {'covid': './covid'},
      install_requires = ['numpy',
                        'scipy',
                        'progressbar',
                        'matplotlib',
                        'pandas',
                        'sphinx-argparse']
)
