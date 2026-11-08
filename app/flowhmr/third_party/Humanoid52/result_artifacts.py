"""Shim: toyxyz artifact bookkeeping -> local compat implementations."""
from ._compat import file_stat

def begin_update(*args, **kwargs):
    return None

def commit_artifacts(*args, **kwargs):
    return None

__all__ = ['file_stat', 'begin_update', 'commit_artifacts']
