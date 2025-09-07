from setuptools import setup, find_packages
import setuptools
import subprocess
import os

setup(
    name='visual_debugger',  # Package name
    version='0.4.0',  # Version of your package
    author='Enes Kuzucu',  # Your name

    description='A module to help debugging visual workflows',  # Short description
    long_description=open('README.md').read(),  # Long description from a README file
    long_description_content_type='text/markdown',  # Type of the long description
#     url='https://github.com/karaposu/image-input-handler',  # URL to the repository
    packages=find_packages(),  # Automatically find packages in the directory
    install_requires=[
        'numpy', 'opencv-python', 'image_input_handler'  # List of dependencies
    ],
    classifiers=[
        'Development Status :: 3 - Alpha',  # Development status
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',  # License as you choose
        'Programming Language :: Python :: 3',  # Supported Python versions
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.7',  # Minimum version requirement of Python
)
