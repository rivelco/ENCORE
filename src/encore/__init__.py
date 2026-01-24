from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("encore")
except PackageNotFoundError:
    __version__ = "unknown"
