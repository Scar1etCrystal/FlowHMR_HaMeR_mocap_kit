# Copyright (c) OpenMMLab. All rights reserved.
import mmcv

from .version import __version__, short_version


def digit_version(version_str):
    digit_version = []
    for x in version_str.split('.'):
        if x.isdigit():
            digit_version.append(int(x))
        elif x.find('rc') != -1:
            patch_version = x.split('rc')
            digit_version.append(int(patch_version[0]) - 1)
            digit_version.append(int(patch_version[1]))
    return digit_version


mmcv_minimum_version = '1.3.8'
mmcv_maximum_version = '1.5.0'
mmcv_version = digit_version(mmcv.__version__)


import warnings
if not (mmcv_version >= digit_version(mmcv_minimum_version)):
    warnings.warn(f'MMCV=={mmcv.__version__} may be incompatible. '
                  f'Expected mmcv>={mmcv_minimum_version}.')

__all__ = ['__version__', 'short_version']
