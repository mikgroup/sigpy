"""B1-selective RF Pulse Design functions.

"""
import numpy as np

from sigpy.mri.rf import slr as slr
from sigpy.mri.rf.util import dinf

__all__ = ['dzb1rf', 'dzb1gSliderrf']


def dzb1rf(dt=2e-6, tb=4, ptype='st', flip=np.pi / 6, pbw=0.3, pbc=2,
           d1=0.01, d2=0.01, os=8, splitAndReflect=True):
    # design a B1-selective excitation pulse, following Grissom JMR 2014
    # pbw = width of passband in Gauss
    # pbc = center of passband in Gauss

    # calculate beta filter ripple
    [bsf, d1, d2] = slr.calcRipples(ptype, d1, d2)

    # calculate pulse duration
    B = 4257 * pbw
    T = tb / B

    # calculate number of samples in pulse
    n = np.int(np.ceil(T / dt / 2) * 2)

    if pbc == 0:  # we want passband as close to zero as possible.
        # do my own dual-band filter design to minimize interaction
        # between the left and right bands

        # build system matrix
        A = np.exp(1j * 2 * np.pi *
                   np.outer(np.arange(-n * os / 2, n * os / 2),
                            np.arange(-n / 2, n / 2)) / (n * os))

        # build target pattern
        ii = np.arange(-n * os / 2, n * os / 2) / (n * os) * 2
        w = dinf(d1, d2) / tb
        f = np.asarray([0, (1 - w) * (tb / 2), (1 + w)
                        * (tb / 2), n / 2]) / (n / 2)
        d = np.double(np.abs(ii) < f[1])
        ds = np.double(np.abs(ii) > f[2])

        # shift the target pattern to minimum center position
        pbc = np.int(np.ceil((f[2] - f[1]) * n * os / 2 + f[1] * n * os / 2))
        dl = np.roll(d, pbc)
        dr = np.roll(d, -pbc)
        dsl = np.roll(ds, pbc)
        dsr = np.roll(ds, -pbc)

        # build error weight vector
        w = dl + dr + d1 / d2 * np.multiply(dsl, dsr)

        # solve for the dual-band filter
        AtA = A.conj().T @ np.multiply(np.reshape(w, (np.size(w), 1)), A)
        Atd = A.conj().T @ np.multiply(w, dr - dl)
        h = np.imag(np.linalg.pinv(AtA) @ Atd)

    else:  # normal design

        # design filter
        h = slr.dzls(n, tb, d1, d2)

        # dual-band-modulate the filter
        om = 2 * np.pi * 4257 * pbc  # modulation frequency
        t = np.arange(0, n) * T / n - T / 2
        h = 2 * h * np.sin(om * t)

    if splitAndReflect:
        # split and flip fm waveform to improve large-tip accuracy
        dom = np.concatenate((h[n // 2::-1], h, h[n:n // 2:-1])) / 2
    else:
        dom = np.concatenate((0 * h[n // 2::-1], h, 0 * h[n:n // 2:-1]))

    # scale to target flip, convert to Hz
    dom = dom * flip / (2 * np.pi * dt)

    # build am waveform
    om1 = np.concatenate((-np.ones(n // 2), np.ones(n), -np.ones(n // 2)))

    return om1, dom


def dzb1gSliderrf(dt=2e-6, G=5, tb=12, ptype='st', flip=np.pi / 6,
                  pbw=0.5, pbc=2, d1=0.01, d2=0.01, splitAndReflect=True):
    # design B1-selective excitation gSlider pulses,
    # following Grissom JMR 2014
    # pbw = width of passband in Gauss
    # pbc = center of passband in Gauss

    # calculate beta filter ripple
    [bsf, d1, d2] = slr.calcRipples(ptype, d1, d2)
    # if ptype == 'st':
    bsf = flip

    # calculate pulse duration
    B = 4257 * pbw
    T = tb / B

    # calculate number of samples in pulse
    n = np.int(np.ceil(T / dt / 2) * 2)

    # modulation frequency to center profile at pbc gauss
    om = 2 * np.pi * 4257 * pbc
    t = np.arange(0, n) * T / n - T / 2

    om1 = np.zeros((2 * n, G))
    dom = np.zeros((2 * n, G))
    for Gind in range(1, G + 1):
        # design filter
        h = bsf * slr.dzgSliderB(n, G, Gind, tb, d1, d2, np.pi, n // 4)
        # if ptype == 'ex':
        # h = slr.b2rf(h)
        # modulate filter to center and add it to a time-reversed and modulated
        # copy, then take the imaginary part to get an odd filter
        h = np.imag(h * np.exp(1j * om * t) - h[n::-1] * np.exp(1j * -om * t))
        if splitAndReflect:
            # split and flip fm waveform to improve large-tip accuracy
            dom[:, Gind - 1] = \
                np.concatenate((h[n // 2::-1], h, h[n:n // 2:-1])) / 2
        else:
            dom[:, Gind - 1] = \
                np.concatenate((0 * h[n // 2::-1], h, 0 * h[n:n // 2:-1]))
        # build am waveform
        om1[:, Gind - 1] = np.concatenate((-np.ones(n // 2), np.ones(n),
                                           -np.ones(n // 2)))

    # scale to target flip, convert to Hz
    dom = dom / (2 * np.pi * dt)

    return om1, dom


def dzb1Hadamardrf(dt=2e-6, G=8, tb=16, ptype='st', flip=np.pi / 6,
                   pbw=2, pbc=2, d1=0.01, d2=0.01, splitAndReflect=True):
    # design B1-selective excitation gSlider pulses,
    # following Grissom JMR 2014
    # pbw = width of passband in Gauss
    # pbc = center of passband in Gauss

    # calculate beta filter ripple
    [bsf, d1, d2] = slr.calcRipples(ptype, d1, d2)
    # if ptype == 'st':
    bsf = flip

    # calculate pulse duration
    B = 4257 * pbw
    T = tb / B

    # calculate number of samples in pulse
    n = np.int(np.ceil(T / dt / 2) * 2)

    # modulation frequency to center profile at pbc gauss
    om = 2 * np.pi * 4257 * pbc
    t = np.arange(0, n) * T / n - T / 2

    om1 = np.zeros((2 * n, G))
    dom = np.zeros((2 * n, G))
    for Gind in range(1, G + 1):
        # design filter
        h = bsf * slr.dzHadamardB(n, G, Gind, tb, d1, d2, n // 4)
        # if ptype == 'ex':
        # h = slr.b2rf(h)
        # modulate filter to center and add it to a time-reversed and modulated
        # copy, then take the imaginary part to get an odd filter
        h = np.imag(h * np.exp(1j * om * t) - h[n::-1] * np.exp(1j * -om * t))
        if splitAndReflect:
            # split and flip fm waveform to improve large-tip accuracy
            dom[:, Gind - 1] = \
                np.concatenate((h[n // 2::-1], h, h[n:n // 2:-1])) / 2
        else:
            dom[:, Gind - 1] = \
                np.concatenate((0 * h[n // 2::-1], h, 0 * h[n:n // 2:-1]))
        # build am waveform
        om1[:, Gind - 1] = np.concatenate((-np.ones(n // 2), np.ones(n),
                                           -np.ones(n // 2)))

    # scale to target flip, convert to Hz
    dom = dom / (2 * np.pi * dt)

    return om1, dom
