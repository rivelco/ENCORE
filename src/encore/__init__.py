from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("encore-toolkit")
except PackageNotFoundError:
    __version__ = "unknown"
