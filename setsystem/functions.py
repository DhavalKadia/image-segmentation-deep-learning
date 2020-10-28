import os

def install_compatible_version(package, min_version):
    if is_installed(package):
        curr_version = get_version(package)

        from packaging import version

        is_compatible = version.parse(min_version) == version.parse(curr_version)

        if not is_compatible:
            os.system("pip uninstall " + package)
            os.system("pip install " + package + "==" + min_version)
    else:
        os.system("pip install " + package + "==" + min_version)

def get_version(package):
    from importlib_metadata import version
    curr_version = version(package)

    return curr_version

def try_import(package):
    try:
        __import__(package)
    except ImportError:
        os.system("pip install " + package)

def is_installed(package):
    try:
        __import__(package)
        return True
    except ImportError:
        return False

def uninstall(package):
    if is_installed(package):
        os.system("pip uninstall " + package)
