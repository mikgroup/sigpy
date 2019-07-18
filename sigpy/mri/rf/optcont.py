"""Optimal Control Pulse Design functions.

"""
import numpy as np

__all__ = ['blochsim', 'deriv']


def blochsim(rf, x, g):

    # assume x has inverse spatial units of g, and g has gamma*dt applied
    # assume x = [...,Ndim], g = [Ndim,Nt]

    a = np.ones(np.shape(x)[0], dtype=complex)
    b = np.zeros(np.shape(x)[0], dtype=complex)
    for mm in range(0, np.size(rf), 1):  # loop over time

        # apply RF
        C = np.cos(np.abs(rf[mm])/2)
        S = 1j*np.exp(1j*np.angle(rf[mm]))*np.sin(np.abs(rf[mm])/2)
        at = a*C - b*np.conj(S)
        bt = a*S + b*C
        a = at
        b = bt

        # apply gradient
        if g.ndim > 1:
            z = np.exp(-1j*x@g[mm, :])
        else:
            z = np.exp(-1j*x*g[mm])
        b = b*z

    # apply total phase accrual
    if g.ndim > 1:
        z = np.exp(1j/2*x@np.sum(g, 0))
    else:
        z = np.exp(1j/2*x*np.sum(g))
    a = a*z
    b = b*z

    return a, b


def deriv(rf, x, g, auxa, auxb, af, bf):

    drf = np.zeros(np.shape(rf), dtype=complex)
    ar = np.ones(np.shape(af), dtype=complex)
    br = np.zeros(np.shape(bf), dtype=complex)

    for mm in range(np.size(rf)-1, -1, -1):

        # calculate gradient blip phase
        if g.ndim > 1:
            z = np.exp(1j/2*x@g[mm, :])
        else:
            z = np.exp(1j/2*x*g[mm])

        # strip off gradient blip from forward sim
        af = af*np.conj(z)
        bf = bf*z

        # add gradient blip to backward sim
        ar = ar*z
        br = br*z

        # strip off the curent rf rotation from forward sim
        C = np.cos(np.abs(rf[mm])/2)
        S = 1j*np.exp(1j*np.angle(rf[mm]))*np.sin(np.abs(rf[mm])/2)
        at = af*C + bf*np.conj(S)
        bt = -af*S + bf*C
        af = at
        bf = bt

        # calculate derivatives wrt rf[mm]
        db1 = np.conj(1j/2*br*bf)*auxb
        db2 = np.conj(1j/2*af)*ar*auxb
        drf[mm] = np.sum(db2 + np.conj(db1))
        if auxa is not None:
            da1 = np.conj(1j/2*bf*ar)*auxa
            da2 = 1j/2*np.conj(af)*br*auxa
            drf[mm] += np.sum(da2 + np.conj(da1))

        # add current rf rotation to backward sim
        art = ar*C - np.conj(br)*S
        brt = br*C + np.conj(ar)*S
        ar = art
        br = brt

    return drf
