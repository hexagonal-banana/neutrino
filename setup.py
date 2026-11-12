import sys
import os
import subprocess
from setuptools import setup, find_packages
from setuptools.command.install import install

class BuildNeutrino(install):
    def run(self):
        CURDIR = os.path.dirname(os.path.realpath(__file__))
        # run the building script 
        subprocess.check_output([sys.executable, os.path.join(CURDIR, "neutrino", "build.py")])
        install.run(self)


setup(
    name='neutrino',  
    version='0.1.0',  
    packages=find_packages(),  
    # package_dir={'': 'neutrino'},  # Set src as the root for packages
    package_data={'build': ['*'], 'tools': ['*']},
    include_package_data=True,  # Include files specified in MANIFEST.in
    install_requires=[
        'toml',
    ],
    py_modules=["neutrino"],
    entry_points={
        'console_scripts': [
            'neutrino-nvcc = neutrino.nvcc_wrapper:main',
            'neutrino = neutrino.cli:main',  # Links 'myentry' command to `main` function in `myentry.py`
        ],
    },
    author='Neutrino Team',  # Anonymous Name
    author_email='anonymous@example.com',  # Anonymous Email
    description='Something',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/neutrino-gpu/neutrino',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License', 
        'Operating System :: POSIX :: Linux',
    ],
    cmdclass={'install': BuildNeutrino},
    python_requires='>=3.10',  # Specify the Python version requirement
    setup_requires=[
        'toml',
    ],
)