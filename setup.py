from distutils.core import setup
import glob
import os

# scripts
scripts = glob.glob(os.path.join('scripts', '*.py'))


setup(
    name='bbbd',

    version='0.1',

    packages=['bbbd',
              'bbbd/util',
              'bbbd/instrument_specific',
              'bbbd/statistic',
              'bbbd/fit_functions'],

    url='https://github.com/giacomov/bbbd',

    license='BSD-3',

    author='Giacomo Vianello',

    author_email='giacomov@stanford.edu',

    description='Bayesian Blocks Burst Detector',

    install_requires=['numexpr',
                      'numpy',
                      'scipy>=0.18',
                      'astropy'],

    scripts=scripts,
)
