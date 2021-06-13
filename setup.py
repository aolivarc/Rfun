"""
This file is part of Rfun, a toolbox for the analysis of teleseismic receiver
functions.

Copyright (C) 2020-2021 Andrés Olivar-Castaño

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
For questions, bug reports, or to suggest new features, please contact me at
olivar.ac@gmail.com.
"""

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
