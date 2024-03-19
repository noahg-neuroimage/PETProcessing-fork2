from setuptools import setup, find_packages

setup(name='pet_cli', version='0.0.1', packages=find_packages(),
      install_requires=['numpy', 'scipy', 'numba', 'pandas', 'nibabel', 'antspyx', 'SimpleITK'],
      entry_points={'console_scripts': ['pet-cli-bids = pet_cli.bids_cli:main',
                                        'pet-cli-tac-interpolate = pet_cli.cli_tac_interpolation:main',
                                        'pet-cli-graph-plot = pet_cli.cli_graphical_plots:main'], }, )
