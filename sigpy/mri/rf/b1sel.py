# -*- coding: utf-8 -*-
""":math:`B_1^{+}`-selective RF Pulse Design functions.

"""
import numpy as np
from sigpy.mri.rf import slr as slr
from sigpy.mri.rf.util import dinf
__all__ = ['dz_b1_rf', 'dz_b1_gslider_rf', 'dz_b1_hadamard_rf']


def dz_b1_rf(dt=2e-6, tb=4, ptype='st', flip=np.pi / 6, pbw=0.3,
             pbc=2, d1=0.01, d2=0.01, os=8, split_and_reflect=True):
    """Design a :math:`B_1^{+}`-selective excitation pulse following Grissom \
    JMR 2014

    Args:
        dt (float): hardware sampling dwell time in s.
        tb (int): time-bandwidth product.
        ptype (string): pulse type, 'st' (small-tip excitation), 'ex' (pi/2
            excitation pulse), 'se' (spin-echo pulse), 'inv' (inversion), or
            'sat' (pi/2 saturation pulse).
        flip (float): flip angle, in radians.
        pbw (float): width of passband in Gauss.
        pbc (float): center of passband in Gauss.
        d1 (float): passband ripple level in :math:`M_0^{-1}`.
        d2 (float): stopband ripple level in :math:`M_0^{-1}`.
        os (int): matrix scaling factor.
        split_and_reflect (bool): option to split and reflect designed pulse.

    Split-and-reflect preserves pulse selectivity when scaled to excite large
    tip-angles.

    Returns:
        2-element tuple containing

        - **om1** (*array*): AM waveform.
        - **dom** (*array*): FM waveform (radians/s).

    References:
        Grissom, W., Cao, Z., & Does, M. (2014).
        :math:`B_1^{+}`-selective excitation pulse design using the Shinnar-Le
        Roux algorithm. Journal of Magnetic Resonance, 242, 189-196.
    """

    # calculate beta filter ripple
    [_, d1, d2] = slr.calc_ripples(ptype, d1, d2)

    # calculate pulse duration
    b = 4257 * pbw
    pulse_len = tb / b

    # calculate number of samples in pulse
    n = np.int(np.ceil(pulse_len / dt / 2) * 2)

    if pbc == 0:
        # we want passband as close to zero as possible.
        # do my own dual-band filter design to minimize interaction
        # between the left and right bands

        # build system matrix
        A = np.exp(1j * 2 * np.pi *
                   np.outer(np.arange(-n * os / 2, n * os / 2),
                            np.arange(-n / 2, n / 2)) / (n * os))

        # build target pattern
        ii = np.arange(-n * os / 2, n * os / 2) / (n * os) * 2
        w = dinf(d1, d2) / tb
        f = np.asarray([0, (1 - w) * (tb / 2),
                        (1 + w) * (tb / 2),
                        n / 2]) / (n / 2)
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
        t = np.arange(0, n) * pulse_len / n - pulse_len / 2
        h = 2 * h * np.sin(om * t)

    if split_and_reflect:
        # split and flip fm waveform to improve large-tip accuracy
        dom = np.concatenate((h[n // 2::-1], h, h[n:n // 2:-1])) / 2
    else:
        dom = np.concatenate((0 * h[n // 2::-1], h, 0 * h[n:n // 2:-1]))

    # scale to target flip, convert to Hz
    dom = dom * flip / (2 * np.pi * dt)

    # build am waveform
    om1 = np.concatenate((-np.ones(n // 2), np.ones(n), -np.ones(n // 2)))

    return om1, dom


def dz_b1_gslider_rf(dt=2e-6, g=5, tb=12, ptype='st', flip=np.pi / 6,
                     pbw=0.5, pbc=2, d1=0.01, d2=0.01, split_and_reflect=True):
    """Design a :math:`B_1^{+}`-selective excitation gSlider pulse following
     Grissom JMR 2014.

    Args:
        dt (float): hardware sampling dwell time in s.
        g (int): number of slabs to be acquired.
        tb (int): time-bandwidth product.
        ptype (string): pulse type, 'st' (small-tip excitation), 'ex' (pi/2
            excitation pulse), 'se' (spin-echo pulse), 'inv' (inversion), or
            'sat' (pi/2 saturation pulse).
        flip (float): flip angle, in radians.
        pbw (float): width of passband in Gauss.
        pbc (float): center of passband in Gauss.
        d1 (float): passband ripple level in :math:`M_0^{-1}`.
        d2 (float): stopband ripple level in :math:`M_0^{-1}`.
        split_and_reflect (bool): option to split and reflect designed pulse.

    Split-and-reflect preserves pulse selectivity when scaled to excite large
     tip-angles.

    Returns:
        2-element tuple containing

        - **om1** (*array*): AM waveform.
        - **dom** (*array*): FM waveform (radians/s).

    References:
        Grissom, W., Cao, Z., & Does, M. (2014).
        :math:`B_1^{+}`-selective excitation pulse design using the Shinnar-Le
        Roux algorithm. Journal of Magnetic Resonance, 242, 189-196.
    """

    # calculate beta filter ripple
    [_, d1, d2] = slr.calc_ripples(ptype, d1, d2)
    # if ptype == 'st':
    bsf = flip

    # calculate pulse duration
    b = 4257 * pbw
    pulse_len = tb / b

    # calculate number of samples in pulse
    n = np.int(np.ceil(pulse_len / dt / 2) * 2)

    om = 2 * np.pi * 4257 * pbc  # modulation freq to center profile at pbc
    t = np.arange(0, n) * pulse_len / n - pulse_len / 2

    om1 = np.zeros((2 * n, g))
    dom = np.zeros((2 * n, g))
    for gind in range(1, g + 1):
        # design filter
        h = bsf*slr.dz_gslider_b(n, g, gind, tb, d1, d2, np.pi, n // 4)

        # modulate filter to center and add it to a time-reversed and modulated
        # copy, then take the imaginary part to get an odd filter
        h = np.imag(h * np.exp(1j * om * t) - h[n::-1] * np.exp(1j * -om * t))
        if split_and_reflect:
            # split and flip fm waveform to improve large-tip accuracy
            dom[:, gind - 1] = np.concatenate((h[n // 2::-1],
                                               h, h[n:n // 2:-1])) / 2
        else:
            dom[:, gind - 1] = np.concatenate((0 * h[n // 2::-1],
                                              h, 0 * h[n:n // 2:-1]))
        # build am waveform
        om1[:, gind - 1] = np.concatenate((-np.ones(n // 2), np.ones(n),
                                          -np.ones(n // 2)))

    # scale to target flip, convert to Hz
    dom = dom / (2 * np.pi * dt)

    return om1, dom


def dz_b1_hadamard_rf(dt=2e-6, g=8, tb=16, ptype='st', flip=np.pi / 6,
                      pbw=2, pbc=2, d1=0.01, d2=0.01, split_and_reflect=True):
    """Design a :math:`B_1^{+}`-selective Hadamard-encoded pulse following \
     Grissom JMR 2014.
    Args:
        dt (float): hardware sampling dwell time in s.
        g (int): number of slabs to be acquired.
        tb (int): time-bandwidth product.
        ptype (string): pulse type, 'st' (small-tip excitation), 'ex' (pi/2 \
            excitation pulse), 'se' (spin-echo pulse), 'inv' (inversion), or \
            'sat' (pi/2 saturation pulse).
        flip (float): flip angle, in radians.
        pbw (float): width of passband in Gauss.
        pbc (float): center of passband in Gauss.
        d1 (float): passband ripple level in :math:`M_0^{-1}`.
        d2 (float): stopband ripple level in :math:`M_0^{-1}`.
        split_and_reflect (bool): option to split and reflect designed pulse.

    Split-and-reflect preserves pulse selectivity when scaled to excite large
    tip-angles.

    Returns:
        2-element tuple containing

        - **om1** (*array*): AM waveform.
        - **dom** (*array*): FM waveform (radians/s).

    References:
        Grissom, W., Cao, Z., & Does, M. (2014).
        :math:`B_1^{+}`-selective excitation pulse design using the Shinnar-Le
        Roux algorithm. Journal of Magnetic Resonance, 242, 189-196.
    """

    # calculate beta filter ripple
    [_, d1, d2] = slr.calc_ripples(ptype, d1, d2)
    bsf = flip

    # calculate pulse duration
    b = 4257 * pbw
    pulse_len = tb / b

    # calculate number of samples in pulse
    n = np.int(np.ceil(pulse_len / dt / 2) * 2)

    # modulation frequency to center profile at pbc gauss
    om = 2 * np.pi * 4257 * pbc
    t = np.arange(0, n) * pulse_len / n - pulse_len / 2

    om1 = np.zeros((2 * n, g))
    dom = np.zeros((2 * n, g))
    for gind in range(1, g + 1):
        # design filter
        h = bsf*slr.dz_hadamard_b(n, g, gind, tb, d1, d2, n // 4)

        # modulate filter to center and add it to a time-reversed and modulated
        # copy, then take the imaginary part to get an odd filter
        h = np.imag(h * np.exp(1j * om * t) - h[n::-1] * np.exp(1j * -om * t))
        if split_and_reflect:
            # split and flip fm waveform to improve large-tip accuracy
            dom[:, gind - 1] = np.concatenate((h[n // 2::-1],
                                              h,
                                              h[n:n // 2:-1])) / 2
        else:
            dom[:, gind - 1] = np.concatenate((0 * h[n // 2::-1],
                                              h,
                                              0 * h[n:n // 2:-1]))
        # build am waveform
        om1[:, gind - 1] = np.concatenate((-np.ones(n // 2), np.ones(n),
                                          -np.ones(n // 2)))

    # scale to target flip, convert to Hz
    dom = dom / (2 * np.pi * dt)

    return om1, dom
