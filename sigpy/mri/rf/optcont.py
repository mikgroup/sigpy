# -*- coding: utf-8 -*-
"""Optimal Control Pulse Design functions.
"""
from sigpy import backend

__all__ = ["blochsim", "deriv"]


def blochsim(rf, x, g):
    r"""1D RF pulse simulation, with simultaneous RF + gradient rotations.
    Assume x has inverse spatial units of g, and g has gamma*dt applied and
    assume x = [...,Ndim], g = [Ndim,Nt].

     Args:
         rf (array): rf waveform input.
         x (array): spatial locations.
         g (array): gradient waveform.

     Returns:
         array: SLR alpha parameter
         array: SLR beta parameter
    """

    device = backend.get_device(rf)
    xp = device.xp
    with device:
        a = xp.ones(xp.shape(x)[0], dtype=complex)
        b = xp.zeros(xp.shape(x)[0], dtype=complex)
        for mm in range(0, xp.size(rf), 1):  # loop over time
            # apply RF
            c = xp.cos(xp.abs(rf[mm]) / 2)
            s = 1j * xp.exp(1j * xp.angle(rf[mm])) * xp.sin(xp.abs(rf[mm]) / 2)
            at = a * c - b * xp.conj(s)
            bt = a * s + b * c
            a = at
            b = bt

            # apply gradient
            if g.ndim > 1:
                z = xp.exp(-1j * x @ g[mm, :])
            else:
                z = xp.exp(-1j * x * g[mm])
            b = b * z

        # apply total phase accrual
        if g.ndim > 1:
            z = xp.exp(1j / 2 * x @ xp.sum(g, 0))
        else:
            z = xp.exp(1j / 2 * x * xp.sum(g))
        a = a * z
        b = b * z

        return a, b


def deriv(rf, x, g, auxa, auxb, af, bf):
    r"""1D RF pulse simulation, with simultaneous RF + gradient rotations.

    'rf', 'g', and 'x' should have consistent units.

     Args:
         rf (array): rf waveform input.
         x (array): spatial locations.
         g (array): gradient waveform.
         auxa (None or array): auxa
         auxb (array): auxb
         af (array): forward sim a.
         bf( array): forward sim b.

     Returns:
         array: SLR alpha parameter
         array: SLR beta parameter
    """

    device = backend.get_device(rf)
    xp = device.xp
    with device:
        drf = xp.zeros(xp.shape(rf), dtype=complex)
        ar = xp.ones(xp.shape(af), dtype=complex)
        br = xp.zeros(xp.shape(bf), dtype=complex)

        for mm in range(xp.size(rf) - 1, -1, -1):
            # calculate gradient blip phase
            if g.ndim > 1:
                z = xp.exp(1j / 2 * x @ g[mm, :])
            else:
                z = xp.exp(1j / 2 * x * g[mm])

            # strip off gradient blip from forward sim
            af = af * xp.conj(z)
            bf = bf * z

            # add gradient blip to backward sim
            ar = ar * z
            br = br * z

            # strip off the curent rf rotation from forward sim
            c = xp.cos(xp.abs(rf[mm]) / 2)
            s = 1j * xp.exp(1j * xp.angle(rf[mm])) * xp.sin(xp.abs(rf[mm]) / 2)
            at = af * c + bf * xp.conj(s)
            bt = -af * s + bf * c
            af = at
            bf = bt

            # calculate derivatives wrt rf[mm]
            db1 = xp.conj(1j / 2 * br * bf) * auxb
            db2 = xp.conj(1j / 2 * af) * ar * auxb
            drf[mm] = xp.sum(db2 + xp.conj(db1))
            if auxa is not None:
                da1 = xp.conj(1j / 2 * bf * ar) * auxa
                da2 = 1j / 2 * xp.conj(af) * br * auxa
                drf[mm] += xp.sum(da2 + xp.conj(da1))

            # add current rf rotation to backward sim
            art = ar * c - xp.conj(br) * s
            brt = br * c + xp.conj(ar) * s
            ar = art
            br = brt

        return drf
