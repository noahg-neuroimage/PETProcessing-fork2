from setuptools import setup, find_packages

setup(name='pet_cli', version='0.1', packages=find_packages(),
      install_requires=['numpy', 'scipy', 'numba', 'pandas', 'nibabel'],
      entry_points={'console_scripts': ['pet-cli = pet_cli.pet_cli:main', ], }, )
