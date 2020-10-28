"""
Run this file to uninstall required packages
"""

from package.setsystem import *
from package.setsystem.functions import uninstall

for key in BASE_PACKAGES:
    # try:
    uninstall(BASE_PACKAGES[key])
    # except ImportError:
    #     print('error is not printed')

for key in PACKAGES:
    # try:
    uninstall(PACKAGES[key])
    # except ImportError:
    #     print('error is not printed')

for key in GPU_SPECIFIC_PACKAGES:
    if key in PACKAGE_VERSION.keys():
        if GPU_ENV:
            package_name = GPU_SPECIFIC_PACKAGES[key][1]
        else:
            package_name = GPU_SPECIFIC_PACKAGES[key][0]

    # try:
    uninstall(package_name)
    # except ImportError:
    #     print('error is not printed')
