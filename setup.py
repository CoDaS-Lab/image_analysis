# -*- coding: utf-8 -*-
"""
image analysis group tools
"""
from setuptools import setup


setup(
    name='image-analysis',
    version='0.0.1',
    url='https://github.com/CoDaS-Lab/image-analysis',
    license='BSD',
    author='CoDaSLab http://shaftolab.com/',
    author_email='s@sophiaray.info',
    description='A small but fast and easy to use stand-alone template '
                'engine written in pure python.',
    long_description=__doc__,
    zip_safe=False,
    classifiers=[
        'Programming Language :: Python :: 3.5',
    ],
    packages=['image-analysis'],
    install_requires=['scikit-learn>=0.17'],
    extras_require=None,
    include_package_data=True
)
