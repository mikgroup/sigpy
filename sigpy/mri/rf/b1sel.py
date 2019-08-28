import numpy as np
import sigpy as sp
import scipy.signal as signal
import sigpy.mri.rf.slr as slr

def rfsim_hp(rf, gamgdt, xx, dom0dt=0):
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
        z = np.exp(-1j*(xx*gamgdt[ii,] + dom0dt))
        b = b*z

        # apply rf
        C = np.cos(np.abs(rf[ii])/2)
        S = 1j*np.exp(1j*np.angle(rf[ii]))*np.sin(np.abs(rf[ii])/2)
        at = a*C - b*np.conj(S)
        bt = a*S + b*C

        a = at
        b = bt

    z = np.exp(1j/2*(xx*np.sum(gamgdt, axis=0) + Nt*dom0dt))
    a = a*z
    b = b*z

    return a, b

def dzb1rf(dt = 2e-6, tb = 4, ptype = 'st', flip = np.pi/6, pbw = 0.3, pbc = 2,
    d1 = 0.01, d2 = 0.01, os = 8):

    # calculate beta filter ripple
    [bsf, d1, d2] = slr.calcRipples(ptype, d1, d2)

    # calculate pulse duration
    B = 4257*pbw
    T = tb/B

    # calculate number of samples in pulse
    n = np.int(np.ceil(T/dt/2)*2)

    # set up inputs to firls
    w = slr.dinf(d1,d2)/tb
    f = np.asarray([0, (1-w)*(tb/2), (1+w)*(tb/2), n/2])/(n/2)
    m = [1, 1, 0, 0]
    wts = [1, d1/d2]

    if pbc == 0: # design using my own dual band design, to avoid interference
                 # between bands

        # build system matrix
        A = np.exp(1j*2*np.pi*np.asarray(np.outer([np.arange(-n*os/2, n*os/2)],
            [np.arange(-n/2, n/2)]))/(n*os))

        # build target pattern
        ii = np.asarray([np.arange(-n*os/2,n*os/2)])/n/os*2
        ii = ii[:,np.newaxis]
        d = np.double(np.abs(ii) < f[1])
        ds = np.double(np.abs(ii) > f[2])

        # shift target pattern to min center position
        pbc = np.ceil((f[2]-f[1])*n*os/2 + f[1]*n*os/2)
        dl = np.roll(d, pbc)
        dr = np.roll(d, -pbc)
        dsl = np.roll(ds, pbc)
        dsr = np.roll(ds, -pbc)

        # build error weights
        w = (dl or dr) + d1/d2*(dsl and dsr)

        # solve for filter
        h = np.imag(np.linalg.pinv(A.conj().transpose()*w.flatten(1)*A)*
            (A.conj().transpose()*(w*(dr-dl))))

    else: # design using firls + modulation

        # design filter
        h = slr.dzls(n, tb, d1, d2)

        # determine modulation frequency
        om = 2*np.pi*4257*pbc

        # get time vector for modulation
        t = np.arange(0, n)*T/n - T/2

        # apply modultation
        h = 2*h*np.sin(om*t)

    # split and flip fm waveform to improve large-tip accuracy
    dom = np.concatenate((h[n//2::-1], h, h[n:n//2:-1]))/2

    # scale to target flip, convert to Hz
    dom = dom*flip/(2*np.pi*dt)

    # build am waveform
    om1 = np.concatenate((-np.ones(n//2), np.ones(n), -np.ones(n//2)))

    return om1, dom
