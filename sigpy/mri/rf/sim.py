"""RF Pulse Simulation functions.

"""
import numpy as np

__all__ = ['abrm', 'abrmnd', 'abrm_hp']

def abrm(rf, x, balanced = False):

    eps = 1e-16

    # 1D Simulation of the RF pulse, with simultaneous RF + gradient rotations
    g = np.ones(np.size(rf))*2*np.pi/np.size(rf)

    a = np.ones(np.size(x), dtype=complex)
    b = np.zeros(np.size(x), dtype=complex)
    for mm in range(0, np.size(rf), 1):
        om = x*g[mm]
        phi = np.sqrt(np.abs(rf[mm])**2 + om**2) + eps
        n = np.column_stack((np.real(rf[mm])/phi, np.imag(rf[mm])/phi, om/phi))
        av = np.cos(phi/2) - 1j*n[:, 2]*np.sin(phi/2)
        bv = -1j*(n[:, 0] + 1j*n[:, 1])*np.sin(phi/2)
        at = av*a - np.conj(bv)*b
        bt = bv*a + np.conj(av)*b
        a = at
        b = bt

    if balanced: # apply a rewinder
        g = -2*np.pi/2
        om = x*g
        phi = np.abs(om) + eps
        nz = om/phi
        av = np.cos(phi/2) - 1j*nz*np.sin(phi/2)
        a = av*a
        b = np.conj(av)*b

    return a, b


def abrmnd(rf, x, g):

    # assume x has inverse spatial units of g, and g has gamma*dt applied
    # assume x = [...,Ndim], g = [Ndim,Nt]
    eps = 1e-16

    a = np.ones(np.shape(x)[0], dtype=complex)
    b = np.zeros(np.shape(x)[0], dtype=complex)
    for mm in range(0, np.size(rf), 1):
        om = x@g[mm, :]
        phi = np.sqrt(np.abs(rf[mm])**2 + om**2)
        n = np.column_stack((np.real(rf[mm])/(phi+eps),
            np.imag(rf[mm])/(phi+eps), om/(phi+eps)))
        av = np.cos(phi/2) - 1j*n[:, 2]*np.sin(phi/2)
        bv = -1j*(n[:, 0] + 1j*n[:, 1])*np.sin(phi/2)
        at = av*a - np.conj(bv)*b
        bt = bv*a + np.conj(av)*b
        a = at
        b = bt

    return a, b


def abrm_hp(rf, gamgdt, xx, dom0dt=0):
    # rf: rf pulse samples in radians
    # gamgdt: gradient samples in radians/(units of xx)
    # xx: spatial locations
    # dom0dt: off-resonance phase in radians

    Ns = np.shape(xx)
    Ns = Ns[0] # Ns: # of spatial locs
    Nt = np.shape(gamgdt)
    Nt = Nt[0] # Nt: # time points

    a = np.ones((Ns,))
    b = np.zeros((Ns,))

    for ii in np.arange(Nt):
        # apply phase accural
        z = np.exp(-1j*(xx*gamgdt[ii,]+dom0dt))
        b = b*z

        # apply rf
        C = np.cos(np.abs(rf[ii])/2)
        S = 1j*np.exp(1j*np.angle(rf[ii]))*np.sin(np.abs(rf[ii])/2)
        at = a*C - b*np.conj(S)
        bt = a*S + b*C

        a = at
        b = bt

    z = np.exp(1j/2*(xx*np.sum(gamgdt,axis=0)+Nt*dom0dt))
    a = a*z
    b = b*z

    return a, b
