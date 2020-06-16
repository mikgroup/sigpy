# -*- coding: utf-8 -*-
"""RF Pulse Simulation Functions.

"""
from sigpy import backend

__all__ = ['abrm', 'abrm_nd', 'abrm_hp', 'abrm_ptx']


def abrm(rf, x, balanced=False):
    r"""1D RF pulse simulation, with simultaneous RF + gradient rotations.

    Args:
         rf (array): rf waveform input.
         x (array): spatial locations.
         balanced (bool): toggles application of rewinder.

    Returns:
        2-element tuple containing

        - **a** (*array*): SLR alpha parameter.
        - **b** (*array*): SLR beta parameter.

    References:
        Pauly, J., Le Roux, Patrick., Nishimura, D., and Macovski, A.(1991).
        'Parameter Relations for the Shinnar-LeRoux Selective Excitation
        Pulse Design Algorithm'.
        IEEE Transactions on Medical Imaging, Vol 10, No 1, 53-65.
     """

    device = backend.get_device(rf)
    xp = device.xp
    with device:
        eps = 1e-16

        g = xp.ones(xp.size(rf)) * 2 * xp.pi / xp.size(rf)

        a = xp.ones(xp.size(x), dtype=complex)
        b = xp.zeros(xp.size(x), dtype=complex)
        for mm in range(xp.size(rf)):
            om = x * g[mm]
            phi = xp.sqrt(xp.abs(rf[mm]) ** 2 + om ** 2) + eps
            n = xp.column_stack((xp.real(rf[mm]) / phi,
                                 xp.imag(rf[mm]) / phi,
                                 om / phi))
            av = xp.cos(phi / 2) - 1j * n[:, 2] * xp.sin(phi / 2)
            bv = -1j * (n[:, 0] + 1j * n[:, 1]) * xp.sin(phi / 2)
            at = av * a - xp.conj(bv) * b
            bt = bv * a + xp.conj(av) * b
            a = at
            b = bt

        if balanced:  # apply a rewinder
            g = -2 * xp.pi / 2
            om = x * g
            phi = xp.abs(om) + eps
            nz = om / phi
            av = xp.cos(phi / 2) - 1j * nz * xp.sin(phi / 2)
            a = av * a
            b = xp.conj(av) * b

        return a, b


def abrm_nd(rf, x, g):
    r"""N-dim RF pulse simulation

    Assumes that x has inverse spatial units of g, and g has gamma*dt applied.

    Assumes dimensions x = [...,Ndim], g = [Ndim,Nt].

    Args:
         rf (array): rf waveform input.
         x (array): spatial locations.
         g (array): gradient array.

    Returns:
        2-element tuple containing

        - **a** (*array*): SLR alpha parameter.
        - **b** (*array*): SLR beta parameter.

    References:
        Pauly, J., Le Roux, Patrick., Nishimura, D., and Macovski, A.(1991).
        'Parameter Relations for the Shinnar-LeRoux Selective Excitation
        Pulse Design Algorithm'.
        IEEE Transactions on Medical Imaging, Vol 10, No 1, 53-65.
     """

    device = backend.get_device(rf)
    xp = device.xp
    with device:
        eps = 1e-16

        a = xp.ones(xp.shape(x)[0], dtype=complex)
        b = xp.zeros(xp.shape(x)[0], dtype=complex)
        for mm in range(xp.size(rf)):
            om = x @ g[mm, :]
            phi = xp.sqrt(xp.abs(rf[mm]) ** 2 + om ** 2)
            n = xp.column_stack((xp.real(rf[mm]) / (phi + eps),
                                 xp.imag(rf[mm]) / (phi + eps),
                                 om / (phi + eps)))
            av = xp.cos(phi / 2) - 1j * n[:, 2] * xp.sin(phi / 2)
            bv = -1j * (n[:, 0] + 1j * n[:, 1]) * xp.sin(phi / 2)
            at = av * a - xp.conj(bv) * b
            bt = bv * a + xp.conj(av) * b
            a = at
            b = bt

        return a, b


def abrm_hp(rf, gamgdt, xx, dom0dt=0):
    r"""1D RF pulse simulation, with non-simultaneous RF + gradient rotations.

    Args:
        rf (array): rf pulse samples in radians.
        gamdt (array): gradient samples in radians/(units of xx).
        xx (array): spatial locations.
        dom0dt (float): off-resonance phase in radians.

    Returns:
        2-element tuple containing

        - **a** (*array*): SLR alpha parameter.
        - **b** (*array*): SLR beta parameter.

    References:
        Pauly, J., Le Roux, Patrick., Nishimura, D., and Macovski, A.(1991).
        'Parameter Relations for the Shinnar-LeRoux Selective Excitation
        Pulse Design Algorithm'.
        IEEE Transactions on Medical Imaging, Vol 10, No 1, 53-65.
     """

    device = backend.get_device(rf)
    xp = device.xp
    with device:
        Ns = xp.shape(xx)
        Ns = Ns[0]  # Ns: # of spatial locs
        Nt = xp.shape(gamgdt)
        Nt = Nt[0]  # Nt: # time points

        a = xp.ones((Ns,))
        b = xp.zeros((Ns,))

        for ii in xp.arange(Nt):
            # apply phase accural
            z = xp.exp(-1j * (xx * gamgdt[ii, ] + dom0dt))
            b = b * z

            # apply rf
            C = xp.cos(xp.abs(rf[ii]) / 2)
            S = 1j * xp.exp(1j * xp.angle(rf[ii])) * xp.sin(xp.abs(rf[ii]) / 2)
            at = a * C - b * xp.conj(S)
            bt = a * S + b * C

            a = at
            b = bt

        z = xp.exp(1j / 2 * (xx * xp.sum(gamgdt, axis=0) + Nt * dom0dt))
        a = a * z
        b = b * z

        return a, b


def abrm_ptx(b1, x, g, dt, fmap=None, sens=None):
    r"""N-dim RF pulse simulation

    Assumes that x has inverse spatial units of g, and g has gamma*dt applied.

    Assumes dimensions rf = [Nc, Nt], x = [...,Ndim], g = [Ndim,Nt], and
    sens = [Nc, dim, dim].

    Args:
         b1 (array): rf waveform input samples in radians.
         x (array): spatial locations (m).
         g (array): gradient array (mT/m with gamma*dt applied).
         dt (float): hardware dwell time (s).
         fmap (array): off-resonance map (Hz).
         sens (array or None): B1+ sensitivity matrix. If None, creates matrix
            of 1's. Input size [Nc dim dim]


    Returns:
        4-element tuple containing

        - **a** (*array*): SLR alpha parameter.
        - **b** (*array*): SLR beta parameter.
        - **m** (*array*): transverse magnetization.
        - **mz** (*array*): longitudinal magnetization.

    References:
        Pauly, J., Le Roux, Patrick., Nishimura, D., and Macovski, A.(1991).
        'Parameter Relations for the Shinnar-LeRoux Selective Excitation
        Pulse Design Algorithm'.
        IEEE Transactions on Medical Imaging, Vol 10, No 1, 53-65.

        Grissom, W., Xu, D., Kerr, A., Fessler, J. and Noll, D. (2009). 'Fast
        large-tip-angle multidimensional and parallel RF pulse design in MRI'
        IEEE Trans Med Imaging, Vol 28, No 10, 1548-59.
     """

    device = backend.get_device(b1)
    xp = device.xp
    with device:

        gam = 267.522 * 1e6 / 1000  # rad/s/mT

        dim = int(xp.sqrt(x.shape[0]))
        Ns = dim * dim
        Nc = b1.shape[0]
        Nt = b1.shape[1]
        dim = int(xp.sqrt(x.shape[0]))

        if sens is None:
            sens = xp.ones((dim*dim, Nc))
        else:
            sens = xp.transpose(sens)
            sens = xp.reshape(sens, (dim*dim, Nc))

        bxy = sens @ b1
        bz = x @ xp.transpose(g)

        if fmap is not None and xp.sum(xp.abs(fmap)) != 0:
            rep_b0 = xp.repeat(xp.expand_dims(fmap.flatten(),  0), Nt, axis=0)
            bz += xp.transpose(rep_b0/gam*2*xp.pi)

        statea = xp.ones((Ns, 1))
        stateb = xp.zeros((Ns, 1))
        a = xp.ones(xp.shape(x)[0], dtype=complex)
        b = xp.zeros(xp.shape(x)[0], dtype=complex)
        for mm in range(Nt):
            phi = dt*gam*xp.sqrt(xp.abs(bxy[:, mm]) ** 2 + bz[:, mm] ** 2)
            with xp.errstate(divide='ignore'):
                normfact = dt*gam*(phi ** -1)
                normfact[xp.isinf(normfact)] = 0
                nxy = normfact * bxy[:, mm]
                nxy[xp.isinf(nxy)] = 0
            nz = normfact * bz[:, mm]
            nz[xp.isinf(nz)] = 0
            cp = xp.cos(phi/2)
            sp = xp.sin(phi/2)
            alpha = xp.expand_dims(cp + 1j * nz * sp, 1)
            beta = xp.expand_dims(1j * xp.conj(nxy) * sp, 1)

            tmpa = xp.multiply(alpha, statea) + xp.multiply(beta,  stateb)
            tmpb = -xp.conj(beta) * statea + xp.conj(alpha) * stateb

            statea, stateb = tmpa, tmpb

            # NOT returning all states:
            a = statea
            b = -xp.conj(stateb)

        mxy0 = 0 + 1j * 0
        mz0 = 1
        m = mz0 * xp.conj(statea) * stateb
        m += mxy0*xp.conj(statea) ** 2
        m -= xp.conj(mxy0)*(stateb ** 2)
        mz = mz0 * (statea * xp.conj(statea) - stateb * xp.conj(stateb))
        mz += 2 * xp.real(mxy0 * xp.conj(statea)*xp.negative(xp.conj(stateb)))

        return a, b, m, mz
