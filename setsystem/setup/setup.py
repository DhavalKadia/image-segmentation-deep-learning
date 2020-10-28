"""
Run this file to install required packages
"""

from package.setsystem import *
from package.setsystem.functions import try_import, install_compatible_version

for key in BASE_PACKAGES:
    if key in PACKAGE_VERSION.keys():
        install_compatible_version(BASE_PACKAGES[key], PACKAGE_VERSION[key])
    else:
        try_import(BASE_PACKAGES[key])

for key in PACKAGES:
    if key in PACKAGE_VERSION.keys():
        install_compatible_version(PACKAGES[key], PACKAGE_VERSION[key])
    else:
        try_import(PACKAGES[key])

for key in GPU_SPECIFIC_PACKAGES:
    if key in PACKAGE_VERSION.keys():
        if GPU_ENV:
            package_name = GPU_SPECIFIC_PACKAGES[key][1]
        else:
            package_name = GPU_SPECIFIC_PACKAGES[key][0]

        install_compatible_version(package_name, PACKAGE_VERSION[key])
    else:
        try_import(GPU_SPECIFIC_PACKAGES[key])
