"""
Setup script for Adaptive Multi-Modal AI Framework for Self-Regulated Learning
"""

from setuptools import setup, find_packages
import os

# Read the contents of README file
this_directory = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

# Read requirements
with open(os.path.join(this_directory, 'requirements.txt'), encoding='utf-8') as f:
    requirements = [line.strip() for line in f if line.strip() and not line.startswith('#')]

setup(
    name='adaptive-srl-framework',
    version='1.0.0',
    author='Anas AlSobeh, Amani Shatnawi, Ahmad Asfour',
    author_email='anas.alsobeh@siu.edu',
    description='A federated deep reinforcement learning framework for personalized self-regulated learning',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/yourusername/adaptive-srl-framework',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'Intended Audience :: Education',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Education',
    ],
    python_requires='>=3.8',
    install_requires=requirements,
    extras_require={
        'dev': [
            'pytest>=7.0.0',
            'pytest-cov>=3.0.0',
            'black>=22.0.0',
            'flake8>=4.0.0',
            'mypy>=0.950',
        ],
        'docs': [
            'sphinx>=4.5.0',
            'sphinx-rtd-theme>=1.0.0',
        ],
    },
    entry_points={
        'console_scripts': [
            'srl-train=src.training.cli:train',
            'srl-evaluate=src.evaluation.cli:evaluate',
            'srl-generate-data=src.data.cli:generate',
        ],
    },
    include_package_data=True,
    zip_safe=False,
)

