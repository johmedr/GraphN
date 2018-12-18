from setuptools import setup, find_packages
from os import path
from io import open

here = path.abspath(path.dirname(__file__))
with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='graphn',
    version='0.0.1',
    description='Graph Neural Networks with Keras made easy.', 
    long_description=long_description,
    long_description_content_type='text/markdown',
    license='MIT',
    url='https://github.com/yop0/GraphN',
    author='Johan Medrano',
    author_email='medrano@etud.insa-toulouse.fr',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
    ],
    #keywords='deep- setuptools development',  
    packages=find_packages(),
    install_requires=['keras'], 
    #extras_require={},
    package_data={
        'graphn': ['README.md'],
    },
    project_urls={
        'Github':'https://github.com/yop0/GraphN'
    },
)