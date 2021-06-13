import os
from setuptools import setup, find_packages

setup(name='rfun',
      version='1.1',
      description='A GUI-based tool for the computation and analysis of receiver functions',
      url='http://github.com/aolivarc/Rfun',
      author='Andrés Olivar-Castaño',
      license='GNU GPL v3',
      packages=['rfun'],
      include_package_data=True,
      zip_safe=False,
      entry_points={'console_scripts':['rfun = rfun.__init__:start_rfun',]}
     )
