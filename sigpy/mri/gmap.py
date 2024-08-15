# -*- coding: utf-8 -*-
"""Methods for computing the geometry factor (g-map)

Author:
    Zhengguo Tan <zhengguo.tan@gmail.com>
"""
import copy
import numpy as np
from sigpy import backend, util
from sigpy.mri import app


def pseudo_replica(app, y, mps,
                   normalize=True, replicas=10):
    r"""
    Performs multiple reconstruction with the supplied
    mri reconstruction app and measured k-space data y.

    Args:
        app: an app for image reconstruction (e.g. SenseRecon).
        y (array): measured k-space data.
        mps (array): coil sensitivity maps.
        normalize (Boolean): normalize y or not [default:True].
        replicas (integer): numer of replicas to be ran for the calculation of the g map.

    Reference:
        http://hansenms.github.io/sunrise/sunrise2013/
    """
    device = backend.get_device(y)
    xp = device.xp

    if normalize is True:
        with device:
            s = xp.linalg.norm(y)
            y = 1E6 * y / s

    with device:
        yvec = y.flatten()
        ind = yvec.nonzero()
        y1 = yvec[ind]
        rshape = ind[0].shape

    res = []
    for r in range(replicas):

        n = util.randn(rshape, scale=1, dtype=y.dtype, device=device)

        with device:
            yn = y1 + 0.1 * n
            yv = xp.zeros_like(yvec)
            yv[ind] = yn
            ym = yv.reshape(y.shape)

        # cfl.writecfl('test_y_' + str(r), ym)

        # pass noisey y to app
        curr_app = app
        curr_app.y = xp.zeros_like(ym)
        imgn = curr_app.run()

        res.append(imgn)

    res = backend.to_device(xp.array(res))

    sca = np.max(np.abs(res))
    g = np.std(res + sca, axis=0)

    # g = g * util.rss(mps)

    return g, res