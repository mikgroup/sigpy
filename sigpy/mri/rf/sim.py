"""RF Pulse Simulation functions.

"""
from sigpy import backend

__all__ = ['abrm', 'abrmnd', 'abrm_hp']


def abrm(rf, x, balanced = False):

    device = backend.get_device(rf)
    xp = device.xp
    with device:
        eps = 1e-16

        # 1D Simulation of the RF pulse, with simultaneous RF + gradient rotations
        g = xp.ones(xp.size(rf))*2*xp.pi/xp.size(rf)

        a = xp.ones(xp.size(x), dtype=complex)
        b = xp.zeros(xp.size(x), dtype=complex)
        for mm in range(0, xp.size(rf), 1):
            om = x*g[mm]
            phi = xp.sqrt(xp.abs(rf[mm])**2 + om**2) + eps
            n = xp.column_stack((xp.real(rf[mm])/phi, xp.imag(rf[mm])/phi, om/phi))
            av = xp.cos(phi/2) - 1j*n[:, 2]*xp.sin(phi/2)
            bv = -1j*(n[:, 0] + 1j*n[:, 1])*xp.sin(phi/2)
            at = av*a - xp.conj(bv)*b
            bt = bv*a + xp.conj(av)*b
            a = at
            b = bt

        if balanced: # apply a rewinder
            g = -2*xp.pi/2
            om = x*g
            phi = xp.abs(om) + eps
            nz = om/phi
            av = xp.cos(phi/2) - 1j*nz*xp.sin(phi/2)
            a = av*a
            b = xp.conj(av)*b

        return a, b


def abrmnd(rf, x, g):

    # assume x has inverse spatial units of g, and g has gamma*dt applied
    # assume x = [...,Ndim], g = [Ndim,Nt]
    device = backend.get_device(rf)
    xp = device.xp
    with device:
        eps = 1e-16

        a = xp.ones(xp.shape(x)[0], dtype=complex)
        b = xp.zeros(xp.shape(x)[0], dtype=complex)
        for mm in range(0, xp.size(rf), 1):
            om = x@g[mm, :]
            phi = xp.sqrt(xp.abs(rf[mm])**2 + om**2)
            n = xp.column_stack((xp.real(rf[mm])/(phi+eps),
                xp.imag(rf[mm])/(phi+eps), om/(phi+eps)))
            av = xp.cos(phi/2) - 1j*n[:, 2]*xp.sin(phi/2)
            bv = -1j*(n[:, 0] + 1j*n[:, 1])*xp.sin(phi/2)
            at = av*a - xp.conj(bv)*b
            bt = bv*a + xp.conj(av)*b
            a = at
            b = bt

        return a, b


def abrm_hp(rf, gamgdt, xx, dom0dt=0):
    # rf: rf pulse samples in radians
    # gamgdt: gradient samples in radians/(units of xx)
    # xx: spatial locations
    # dom0dt: off-resonance phase in radians
    device = backend.get_device(rf)
    xp = device.xp
    with device:
        Ns = xp.shape(xx)
        Ns = Ns[0] # Ns: # of spatial locs
        Nt = xp.shape(gamgdt)
        Nt = Nt[0] # Nt: # time points

        a = xp.ones((Ns,))
        b = xp.zeros((Ns,))

        for ii in xp.arange(Nt):
            # apply phase accural
            z = xp.exp(-1j*(xx*gamgdt[ii,] + dom0dt))
            b = b*z

            # apply rf
            C = xp.cos(xp.abs(rf[ii])/2)
            S = 1j*xp.exp(1j*xp.angle(rf[ii]))*xp.sin(xp.abs(rf[ii])/2)
            at = a*C - b*xp.conj(S)
            bt = a*S + b*C

            a = at
            b = bt

        z = xp.exp(1j/2*(xx*xp.sum(gamgdt, axis=0) + Nt*dom0dt))
        a = a*z
        b = b*z

        return a, b
