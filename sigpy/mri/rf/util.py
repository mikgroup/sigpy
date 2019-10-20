# -*- coding: utf-8 -*-
"""MRI RF utilities.
"""

import numpy as np

__all__ = ['dinf']


def dinf(d1=0.01, d2=0.01):
    a1 = 5.309e-3
    a2 = 7.114e-2
    a3 = -4.761e-1
    a4 = -2.66e-3
    a5 = -5.941e-1
    a6 = -4.278e-1

    l10d1 = np.log10(d1)
    l10d2 = np.log10(d2)

    d = (a1 * l10d1 * l10d1 + a2 * l10d1 + a3) * l10d2 \
        + (a4 * l10d1 * l10d1 + a5 * l10d1 + a6)

    return d
