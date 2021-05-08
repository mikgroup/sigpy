# -*- coding: utf-8 -*-
"""MRI gradient and excitation trajectory design
"""

import numpy as np
from scipy import interpolate
from scipy import integrate
import numba as nb
import math


__all__ = ['min_trap_grad', 'trap_grad', 'spiral_varden', 'spiral_arch', 'epi',
           'rosette', 'spokes_grad', 'stack_of', 'traj_array_to_complex',
           'traj_complex_to_array', 'min_time_gradient']


def min_trap_grad(area, gmax, dgdt, dt):
    r"""Minimal duration trapezoidal gradient designer. Design for target area
    under the flat portion (for non-ramp-sampled pulses)

    Args:
        area (float): pulse area in (g*sec)/cm
        gmax (float): maximum gradient in g/cm
        dgdt (float): max slew rate in g/cm/sec
        dt (float): sample time in sec

    Returns:
        2-element tuple containing

        - **trap** (*array*): gradient waveform in g/cm.
        - **ramppts** (*int*): number of points in ramps.

    """

    if np.abs(area) > 0:
        # we get the solution for plateau amp by setting derivative of
        # duration as a function of amplitude to zero and solving
        a = np.sqrt(dgdt * area / 2)

        # finish design with discretization
        # make a flat portion of magnitude a and enough area for the swath
        pts = np.floor(area / a / dt)
        flat = np.ones((1, int(pts)))
        flat = flat / np.sum(flat) * area / dt
        if np.max(flat) > gmax:
            flat = np.ones((1, int(np.ceil(area / gmax / dt))))
            flat = flat / np.sum(flat) * area / dt

        # make attack and decay ramps
        ramppts = int(np.ceil(np.max(flat) / dgdt / dt))
        ramp_up = np.linspace(0, ramppts, num=ramppts+1) / ramppts*np.max(flat)
        ramp_dn = np.linspace(ramppts, 0, num=ramppts+1) / ramppts*np.max(flat)

        trap = np.concatenate((ramp_up, np.squeeze(flat), ramp_dn))

    else:
        # negative-area trap requested?
        trap, ramppts = 0, 0

    return np.expand_dims(trap, axis=0), ramppts


def trap_grad(area, gmax, dgdt, dt, *args):
    r"""General trapezoidal gradient designer for total target area
    (for rewinders)

    Args:
        area (float): pulse area in (g*sec)/cm
        gmax (float): maximum gradient in g/cm
        dgdt (float): max slew rate in g/cm/sec
        dt (float): sample time in sec

    Returns:
        2-element tuple containing

        - **trap** (*array*): gradient waveform in g/cm.
        - **ramppts** (*int*): number of points in ramps.

    """

    if len(args) < 5:
        # in case we are making a rewinder
        rampsamp = 1

    if np.abs(area) > 0:
        if rampsamp:

            ramppts = int(np.ceil(gmax/dgdt/dt))
            triareamax = ramppts * dt * gmax

            if triareamax > np.abs(area):
                # triangle pulse
                newgmax = np.sqrt(np.abs(area) * dgdt)
                ramppts = int(np.ceil(newgmax/dgdt/dt))
                ramp_up = np.linspace(0, ramppts, num=ramppts+1)/ramppts
                ramp_dn = np.linspace(ramppts, 0, num=ramppts+1)/ramppts
                pulse = np.concatenate((ramp_up, ramp_dn))
            else:
                # trapezoid pulse
                nflat = int(np.ceil((area - triareamax)/gmax / dt / 2) * 2)
                ramp_up = np.linspace(0, ramppts, num=ramppts+1) / ramppts
                ramp_dn = np.linspace(ramppts, 0, num=ramppts+1) / ramppts
                pulse = np.concatenate((ramp_up, np.ones(nflat), ramp_dn))

            trap = pulse * (area / (sum(pulse) * dt))

        else:
            # make a flat portion of magnitude gmax
            # and enough area for the entire swath
            flat = np.ones(1, np.ceil(area/gmax/dt))
            flat = flat / sum(flat) * area / dt
            flat_top = np.max(flat)

            # make attack and decay ramps
            ramppts = int(np.ceil(np.max(flat) / dgdt / dt))
            ramp_up = np.linspace(0, ramppts, num=ramppts+1) / ramppts*flat_top
            ramp_dn = np.linspace(ramppts, 0, num=ramppts+1) / ramppts*flat_top
            trap = np.concatenate((ramp_up, flat, ramp_dn))

    else:
        trap, ramppts = 0, 0

    return np.expand_dims(trap, axis=0), ramppts


def spiral_varden(fov, res, gts, gslew, gamp, densamp, dentrans, nl,
                  rewinder=False):
    r"""Variable density spiral designer. Produces trajectory, gradients,
    and slew rate. Gradient units returned are in g/cm, g/cm/s

    Args:
        fov (float): imaging field of view (cm).
        res (float): imaging isotropic resolution (cm).
        gts (float): gradient sample time in sec.
        gslew (float): max slew rate in g/cm/s.
        gamp (float): max gradient amplitude in g/cm.
        densamp (float):  duration of full density sampling (# of samples).
        dentrans (float): duration of transition from higher to lower
            (should be >= densamp/2).
        nl (float): degree of undersampling outer region.
        rewinder (Boolean): if True, include rewinder. If false, exclude.

    Returns:
        tuple: (g, k, t, s, dens) tuple containing

        - **g** - (array): gradient waveform [g/cm]
        - **k** - (array): exact k-space corresponding to gradient g.
        - **time** - (array):  sampled time
        - **s** - (array): slew rate [g/cm/s]
        - **dens** - (array): undersampling factor at each time point.

    References:
        Code and algorithm based on spiralgradlx6 from
        Doug Noll, U. of Michigan BME
    """
    fsgcm = gamp  # fullscale g/cm
    risetime = gamp / gslew * 10000  # us
    ts = gts  # sampling time
    gts = gts  # gradient sampling time
    N = np.floor(fov/res)
    targetk = N / 2
    A = 32766  # output scaling of waveform (fullscale)

    max_dec_ratio = 32
    gam = 4257.0
    S = (gts / 1e-6) * A / risetime
    dr = ts / gts
    OMF = 2.0 * np.pi * fov / (1 / (gam * fsgcm * gts))
    OM = 2.0 * np.pi / nl * fov / (1 / (gam * fsgcm * gts))
    distance = 1.0 / (fov * gam * fsgcm * gts / A)

    ac = A
    loop = 1
    absk = 0
    dec_ratio = 1
    s0 = gslew * 100
    ggx, ggy = [], []
    dens = []
    kx, ky = [], []

    while loop > 0:
        loop = 0
        om = OM / dec_ratio
        omf = OMF / dec_ratio
        s = S / dec_ratio
        g0 = 0
        gx = g0
        gy = 0
        absg = np.abs(g0)
        oldkx = 0
        oldky = 0
        tkx = gx
        tky = gy
        kxt = tkx
        kyt = tky
        thetan_1 = 0
        taun = 0
        n = 0
        den1 = 0

        while absk < targetk:
            realn = n / dec_ratio
            taun_1 = taun
            taun = np.abs(tkx + 1j * tky) / A
            tauhat = taun
            if realn > densamp:
                if den1 == 0:
                    den1 = 1

                if realn > densamp + dentrans:
                    if 'scthat' not in locals():
                        scthat = 0
                    scoffset = scthat
                    denoffset = taun_1
                    scthat = scoffset + om * (tauhat - denoffset)
                    fractrans = 1

                else:
                    scoffset = scthat
                    denoffset = taun_1
                    fractrans = (realn - densamp) / dentrans
                    fractrans = 1 - ((fractrans - 1) * (fractrans - 1))
                    scthat = (omf + (om - omf) * fractrans)
                    scthat *= (tauhat - denoffset)
                    scthat += scoffset

            else:
                fractrans = 0
                scthat = omf * tauhat

            theta = np.arctan2(scthat, 1.0) + scthat

            if absg < ac:
                deltheta = theta - thetan_1
                B = 1.0 / (1.0 + np.tan(deltheta) * np.tan(deltheta))
                gtilde = absg
                t1 = s * s
                t2 = gtilde * gtilde * (1 - B)

                if t2 > t1:
                    dec_ratio = dec_ratio * 2.0

                    if dec_ratio > max_dec_ratio:
                        print('k-space calculation failed.\n')
                        return

                    loop = 1
                    break

                t3 = np.sqrt(t1 - t2)
                absg = np.sqrt(B) * gtilde + t3

                if absg > ac:
                    absg = ac

            tgx = absg * np.cos(theta)
            tgy = absg * np.sin(theta)
            tkx += tgx
            tky += tgy
            thetan_1 = theta

            if np.remainder(n, dec_ratio) == 0:
                m = int(np.round(n / dec_ratio))
                gx = np.round((tkx - oldkx) / dec_ratio)
                gx = gx - np.remainder(gx, 2)
                gy = np.round((tky - oldky) / dec_ratio)
                gy = gy - np.remainder(gy, 2)
                if m > len(ggx) - 1:
                    ggx.append(gx)
                    ggy.append(gy)
                else:
                    ggx[m] = gx
                    ggy[m] = gy
                kxt = kxt + gx
                kyt = kyt + gy
                oldkx = tkx
                oldky = tky

                if np.remainder(m, dr) == 0:
                    m = int(m / dr)
                    absk = np.abs(kxt + 1j * kyt) / distance

                    if m > len(dens) - 1:
                        dens.append(omf / (omf + (om - omf) * fractrans))
                        if absk > targetk:
                            break
                        kx.append(kxt / distance)
                        ky.append(kyt / distance)
                    else:
                        dens[m] = omf / (omf + (om - omf) * fractrans)
                        if absk > targetk:
                            break
                        kx[m] = kxt / distance
                        ky[m] = kyt / distance

            n += 1

    g = []
    for i in range(len(ggx)):
        g.append(complex(ggx[i], ggy[i]) / A * fsgcm)
    dt = gts * 1000
    delk = 1 / 4.258 / fov  # (g ms)/cm

    # ramp down
    l2 = len(g) - 1
    rsteps = int(np.ceil(np.abs(g[l2]) / (s0 * 0.99) / gts))
    ind3 = l2 + np.linspace(1, rsteps, num=rsteps)
    c = g[l2] * np.linspace(rsteps, 0, num=rsteps) / rsteps
    g.extend(c)
    dens.extend([0] * len(ind3))

    # rewinder
    if rewinder:
        rewx, ramppts = np.squeeze(trap_grad(abs(np.real(sum(g))) * gts,
                                   gamp, gslew * 50, gts))
        rewy, ramppts = np.squeeze(trap_grad(abs(np.imag(sum(g))) * gts,
                                   gamp, gslew * 50, gts))

        # append rewinder gradient
        if len(rewx) > len(rewy):
            r = -np.sign(np.real(sum(g))) * rewx
            p = np.sign(np.imag(sum(g)))
            p *= 1j * np.abs(np.imag(sum(g))) / np.real(sum(g)) * rewx
            r -= p
        else:
            p = -np.sign(np.real(sum(g)))
            p *= np.abs(np.real(sum(g)) / np.imag(sum(g))) * rewy
            r = p - 1j * np.sign(np.imag(sum(g))) * rewy

        g = np.concatenate((g, r))

    # change from (real, imag) notation to (Nt, 2) notation
    gtemp = np.zeros((len(g), 2))
    gtemp[:, 0] = np.real(g)
    gtemp[:, 1] = np.imag(g)
    g = gtemp

    # calculate trajectory, slew rate factor from designed gradient
    k = np.cumsum(g, axis=0) * dt / delk / fov  # trajectory
    t = np.linspace(0, len(g), num=len(g) + 1)  # time vector
    s = np.diff(g, axis=0) / (gts * 1000)  # slew rate factor

    return g, k, t, s, dens


def spiral_arch(fov, res, gts, gslew, gamp):
    r"""Analytic Archimedean spiral designer. Produces trajectory, gradients,
    and slew rate. Gradient returned has units mT/m.

    Args:
        fov (float): imaging field of view in m.
        res (float): resolution, in m.
        gts (float): sample time in s.
        gslew (float): max slew rate in mT/m/ms.
        gamp (float): max gradient amplitude in mT/m.

    Returns:
        tuple: (g, k, t, s) tuple containing

        - **g** - (array): gradient waveform [mT/m]
        - **k** - (array): exact k-space corresponding to gradient g.
        - **time** - (array):  sampled time
        - **s** - (array): slew rate [mT/m/ms]

    References:
        Glover, G. H.(1999).
        Simple Analytic Spiral K-Space Algorithm.
        Magnetic resonance in medicine, 42, 412-415.

        Bernstein, M.A.; King, K.F.; amd Zhou, X.J. (2004).
        Handbook of MRI Pulse Sequences. Elsevier.
    """

    gam = 267.522 * 1e6 / 1000  # rad/s/mT
    gambar = gam / 2 / np.pi  # Hz/mT
    N = int(fov / res)  # effective matrix size
    lam = 1 / (2 * np.pi * fov)
    beta = gambar * gslew / lam

    kmax = N / (2 * fov)
    a_2 = (9 * beta / 4) ** (1 / 3)  # rad ** (1/3) / s ** (2/3)
    lamb = 5
    theta_max = kmax / lam
    ts = (3 * gam * gamp / (4 * np.pi * lam * a_2 ** 2)) ** 3
    theta_s = 0.5 * beta * ts ** 2
    theta_s /= (lamb + beta / (2 * a_2) * ts ** (4 / 3))
    t_g = np.pi * lam * (theta_max ** 2 - theta_s ** 2) / (gam * gamp)
    n_s = int(np.round(ts / gts))
    n_g = int(np.round(t_g / gts))

    if theta_max > theta_s:
        print(' Spiral trajectory is slewrate limited or amplitude limited')

        tacq = ts + t_g

        t_s = np.linspace(0, ts, n_s)
        t_g = np.linspace(ts + gts, tacq, n_g)

        theta_1 = beta / 2 * t_s ** 2
        theta_1 /= (lamb + beta / (2 * a_2) * t_s ** (4 / 3))
        theta_2 = theta_s ** 2 + gam / (np.pi * lam) * gamp * (t_g - ts)
        theta_2 = np.sqrt(theta_2)

        k1 = lam * theta_1 * (np.cos(theta_1) + 1j * np.sin(theta_1))
        k2 = lam * theta_2 * (np.cos(theta_2) + 1j * np.sin(theta_2))
        k = np.concatenate((k1, k2), axis=0)

    else:

        tacq = 2 * np.pi * fov / 3 * np.sqrt(np.pi / (gam * gslew * res ** 3))
        n_t = int(np.round(tacq / gts))
        t_s = np.linspace(0, tacq, n_t)
        theta_1 = beta / 2 * t_s ** 2
        theta_1 /= (lamb + beta / (2 * a_2) * t_s ** (4 / 3))

        k = lam * theta_1 * (np.cos(theta_1) + 1j * np.sin(theta_1))

    # end of trajectory calculation; prepare outputs
    g = np.diff(k, 1, axis=0) / (gts * gambar)  # gradient
    g = np.pad(g, (0, 1), 'constant')
    s = np.diff(g, 1, axis=0) / (gts * 1000)  # slew rate factor
    s = np.pad(s, (0, 1), 'constant')

    # change from (real, imag) notation to (Nt, 2) notation
    k = traj_complex_to_array(k)
    g = traj_complex_to_array(g)
    s = traj_complex_to_array(s)

    t = np.linspace(0, len(g), num=len(g) + 1)  # time vector

    return g, k, t, s


def epi(fov, n, etl, dt, gamp, gslew, offset=0, dirx=-1, diry=1):
    r"""Basic EPI single-shot trajectory designer.

    Args:
        fov (float): imaging field of view in cm.
        n (int): # of pixels (square). N = etl*nl, where etl = echo-train-len
            and nl = # leaves (shots). nl default 1.
        etl (int): echo train length.
        dt (float): sample time in s.
        gamp (float): max gradient amplitude in mT/m.
        gslew (float): max slew rate in mT/m/ms.
        offset (int): used for multi-shot EPI goes from 0 to #shots-1
        dirx (int): x direction of EPI -1 left to right, 1 right to left
        diry (int): y direction of EPI -1 bottom-top, 1 top-bottom

    Returns:
        tuple: (g, k, t, s) tuple containing

        - **g** - (array): gradient waveform [mT/m]
        - **k** - (array): exact k-space corresponding to gradient g.
        - **time** - (array):  sampled time
        - **s** - (array): slew rate [mT/m/ms]


    References:
        From Antonis Matakos' contrib to Jeff Fessler's IRT.
    """
    s = gslew * dt * 1000

    scaley = 20

    # make the various gradient waveforms
    gamma = 4.2575  # kHz/Gauss
    g = (1 / (1000 * dt)) / (gamma * fov)  # Gauss/cm
    if g > gamp:
        g = gamp
        print('max g reduced to {}'.format(g))

    # readout trapezoid
    gxro = g * np.ones((1, n))  # plateau of readout trapezoid
    areapd = np.sum(gxro) * dt

    ramp = np.expand_dims(np.linspace(s, g, int(g/s)), axis=0)
    gxro = np.concatenate((np.expand_dims(np.array([0]), axis=1), ramp, gxro,
                           np.fliplr(ramp)), axis=1)

    # x prewinder. make sure res_kpre is even. Handle even N by changing prew.
    if n % 2 == 0:
        area = (np.sum(gxro) - dirx * g) * dt
    else:
        area = np.sum(gxro) * dt
    gxprew = dirx * trap_grad(area / 2, gamp, gslew * 1000, dt)[0]

    gxprew = np.concatenate((np.zeros((1, (gxprew.size + ramp.size) % 2)),
                            gxprew), axis=1)

    # partial dephaser (one cycle of phase across each voxel)
    gxpd = -trap_grad(areapd / 2, gamp, gslew * 1000, dt)[0]
    gxpd = np.concatenate((np.zeros((1, gxpd.size % 2)), gxpd), axis=1)

    # phase-encode trapezoids before/after gx
    # handle even N by changing prewinder
    if n % 2 == 0:
        areayprew = areapd / 2 - offset * g * dt
    else:
        areayprew = (areapd - g * dt) / 2 - offset * g * dt

    gyprew = diry * trap_grad(areayprew, gamp, gslew / scaley * 1000, dt)[0]
    gyprew = np.concatenate((np.zeros((1, gyprew.size % 2)), gyprew), axis=1)

    lx = gxpd.size
    ly = gyprew.size
    if lx > ly:
        gyprew = np.concatenate((gyprew, np.zeros((1, lx - ly))), axis=1)
    else:
        gxpd = np.concatenate((gxpd, np.zeros((1, ly - lx))), axis=1)

    # gy readout gradient elements
    # changed readout patterns to create interleaved EPIs
    areagyblip = areapd / etl
    gyblip = trap_grad(areagyblip, gamp, gslew / scaley * 1000, dt)[0]
    gyro = np.concatenate((np.zeros((1, gxro.size - gyblip.size)), gyblip),
                          axis=1)
    gyro2 = np.expand_dims(np.array([0]), axis=1)

    # put together gx and gy

    gxro = -dirx * gxro
    gx = gxprew

    gyro = -diry * gyro
    gyro2 = -diry * gyro2
    gy = np.expand_dims(np.array([0]), axis=1)
    lx = gx.size
    ly = gy.size
    if lx > ly:
        gy = np.concatenate((gy, np.zeros((1, lx - ly))), axis=1)
    else:
        gx = np.concatenate((gx, np.zeros((1, ly - lx))), axis=1)

    gy = np.concatenate((gy, np.zeros((1, int(gyblip.size/2)))), axis=1)

    for ee in range(1, etl):
        flip = ((-1) ** (ee + 1))
        gx = np.concatenate((gx,  flip * gxro), axis=1)
        gy = np.concatenate((gy, gyro), axis=1)

    if etl == 1:
        ee = 1
    else:
        ee += 1

    # concatenate with added 0 to limit max s
    gx = np.concatenate((gx, (-1 ** (ee + 1) * gxro),
                         np.expand_dims(np.array([0]), axis=1)), axis=1)
    gy = np.concatenate((gy, np.zeros((1, gx.size - gy.size))), axis=1)

    # add rephasers at end of gx and gy readout
    areagx = np.sum(gx) * dt
    gxrep = trap_grad(-areagx, gamp, gslew * 1000, dt)[0]
    gx = np.concatenate((gx, gxrep), axis=1)

    areagy = np.sum(gy) * dt  # units = G/cm*s
    gyrep = trap_grad(-areagy, gamp, gslew / scaley * 1000, dt)[0]
    gy = np.concatenate((gy, gyrep), axis=1)

    # make sure length of gx and gy are same, and even
    lx = gx.size
    ly = gy.size
    if lx > ly:
        gy = np.concatenate((gy, np.zeros((1, lx - ly))), axis=1)
    else:
        gx = np.concatenate((gx, np.zeros((1, ly - lx))), axis=1)

    gx = np.concatenate((gx, np.zeros((1, gx.size % 2))), axis=1)
    gy = np.concatenate((gy, np.zeros((1, gy.size % 2))), axis=1)
    g = np.concatenate((gx, gy), axis=0)

    sx = np.diff(gx, axis=1) / (dt * 1000)
    sy = np.diff(gy, axis=1) / (dt * 1000)
    s = np.concatenate((sx, sy), axis=0)

    kx = np.cumsum(gx, axis=1) * gamma * dt * 1000
    ky = np.cumsum(gy, axis=1) * gamma * dt * 1000
    k = np.concatenate((kx, ky), axis=0)

    t = np.linspace(0, kx.size, kx.size) * dt

    return g, k, t, s


def rosette(kmax, w1, w2, dt, dur, gamp=None, gslew=None):
    r"""Basic rosette trajectory designer.

    Args:
        kmax (float): 1/m.
        w1 (float): rotational frequency (Hz).
        w2 (float): center sampling frequency (Hz).
        dt (float): sample time (s).
        dur (float): total duration (s).
        gamp (float): max gradient amplitude (mT/m).
        gslew (float): max slew rate (mT/m/ms).

    Returns:
        tuple: (g, k, t, s) tuple containing

        - **g** - (array): gradient waveform [mT/m]
        - **k** - (array): exact k-space corresponding to gradient g.
        - **time** - (array):  sampled time
        - **s** - (array): slew rate [mT/m/ms]

    References:
        D. C. Noll, 'Multi-shot rosette trajectories for spectrally selective
        MR imaging.' IEEE Trans. Med Imaging 16, 372-377 (1997).
    """

    # check if violates gradient or slew rate constraints
    gam = 267.522 * 1e6 / 1000  # rad/s/mT
    gambar = gam / 2 / np.pi  # Hz/mT
    if gamp is not None:
        if (1 / gambar) * kmax * w1 > gamp:
            print("gmax exceeded, decrease rosette kmax or w1")
            return
    if gslew is not None:
        if (1 / gambar) * kmax * (w1 ** 2 + w2 ** 2) / 1000 > gslew:
            print("smax exceeded, dcrease rosette kmax, w1, or w2")
            return
    t = np.linspace(0, dur, dur / dt)
    k = kmax * np.sin(w1 * t) * np.exp(1j * w2 * t)

    # end of trajectory calculation; prepare outputs
    g = np.diff(k, 1, axis=0) / (dt * gambar)  # gradient
    g = np.pad(g, (0, 1), 'constant')
    s = np.diff(g, 1, axis=0) / (dt * 1000)  # slew rate factor
    s = np.pad(s, (0, 1), 'constant')

    # change from (real, imag) notation to (Nt, 2) notation
    k = traj_complex_to_array(k)
    g = traj_complex_to_array(g)
    s = traj_complex_to_array(s)

    t = np.linspace(0, len(g), num=len(g) + 1)  # time vector

    return g, k, t, s


def spokes_grad(k, tbw, sl_thick, gmax, dgdtmax, gts):
    r""" Spokes gradient designer. Given some chosen spoke locations k, return
    the gradients required to move between those spoke locations.

    Args:
        k (array): spokes locations, [Nspokes, 2]
        tbw (int): time bandwidth product.
        sl_thick (float): slice thickness (mm).
        gmax (float): max gradient amplitude (g/cm).
        dgdtmax (float): max gradient slew (g/cm/s).
        gts (float): hardware sampling dwell time (s).

    Returns:
        g (array): gz, gy, and gz waveforms  in g/cm [3, Nt]

    References:
           Grissom, W., Khalighi, M., Sacolick, L., Rutt, B. & Vogel, M (2012).
           Small-tip-angle spokes pulse design using interleaved greedy and
           local optimization methods. Magnetic Resonance in Medicine, 68(5),
           1553-62.

    """
    n_spokes = k.shape[0]

    area = tbw / (sl_thick / 10) / 4257  # thick * kwid = twb, kwid = gam*area
    [subgz, nramp] = min_trap_grad(area, gmax, dgdtmax, gts)

    # calc gradient, add extra 0 location at end for return to (0, 0)
    gxarea = np.diff(np.concatenate((k[:, 0], np.zeros(1)))) / 4257
    gyarea = np.diff(np.concatenate((k[:, 1], np.zeros(1)))) / 4257

    gx, gy, gz = [], [], []
    gz_sign = -1
    for ii in range(n_spokes):
        gz_sign *= -1
        gz.extend(np.squeeze(gz_sign * subgz).tolist())  # alt sign of gz

        gx.extend([0] * np.size(subgz))  # zeros for gz duration
        if np.absolute(gxarea[ii]) > 0:
            [gblip, _] = trap_grad(abs(gxarea[ii]), gmax, dgdtmax, gts)
            gxblip = np.int(np.sign(gxarea[ii])) * gblip
            gx = gx[:len(gx) - len(gxblip.T)]
            gx.extend(np.squeeze(gxblip).tolist())

        gy.extend([0] * np.size(subgz))
        if np.absolute(gyarea[ii]) > 0:
            [gblip, _] = trap_grad(abs(gyarea[ii]), gmax, dgdtmax, gts)
            gyblip = np.int(np.sign(gyarea[ii])) * gblip
            gy = gy[:len(gy) - len(gyblip.T)]
            gy.extend(np.squeeze(gyblip).tolist())

    [gref, _] = trap_grad(gts * np.sum(subgz) / 2, gmax, dgdtmax, gts)
    gzref = - gref
    gz.extend(np.squeeze(gzref).tolist())
    gx.extend([0] * np.size(gzref))
    gy.extend([0] * np.size(gzref))

    # combine gradient waveforms
    gx = np.array(gx)
    g = np.vstack((np.array(gx), np.array(gy), np.array(gz)))

    return g


def stack_of(k, num, zres):
    r"""Function for creating a 3D stack of ____ trajectory from a 2D [Nt 2]
    trajectory.

    Args:
        k (array): 2D array in [2 x Nt]. Will be bottom of stack.
        num (int): number of layers of stack.
        zres (float): spacing between stacks in cm.
    """

    z = np.linspace(- num * zres / 2, num * zres / 2, num)
    kout = np.zeros((k.shape[0]*num, 3))

    # we will be performing a complex rotation on our trajectory
    k = traj_array_to_complex(k)

    for ii in range(num):
        kr = k[0:] * np.exp(2 * np.pi * 1j * ii / num)
        z_coord = np.expand_dims(np.ones(len(kr)) * z[ii], axis=1)
        krz = np.concatenate((traj_complex_to_array(kr), z_coord), axis=1)

        kout[ii * len(krz):(ii + 1) * len(krz), :] = krz

    return kout


def traj_complex_to_array(k):
    r"""Function to convert complex convention trajectory to [Nt 2] trajectory

    Args:
        k (complex array): Nt vector
    """
    kout = np.zeros((len(k), 2))
    kout[:, 0], kout[:, 1] = np.real(k), np.imag(k)
    return kout


def traj_array_to_complex(k):
    r"""Function to convert [Nt 2] convention traj to complex convention

    Args:
        k (complex array): Nt vector
    """
    kout = k[:, 0] + 1j * k[:, 1]
    return kout


@nb.jit(nopython=True, cache=True)  # pragma: no cover
def runge_kutta(ds: float, st: float, kvals: np.ndarray, smax=None,
                gamma=4.257):
    r"""Runge-Kutta 4 for curve constrained

    Args:
        ds (float): spacing in arc length space
        st (float): output shape.
        kvals (array): 3 points of curve.
        smax (float): maximum slew
        gamma (float): gyromagnetic ratio

    Returns:
        float or None: step size dsdt or None
    """
    temp = (gamma ** 2 * smax ** 2 - abs(kvals[0]) ** 2 * st ** 4)
    if temp < 0.0:
        return None
    k1 = ds / st * math.sqrt(temp)

    temp = \
        (gamma ** 2 * smax ** 2 - abs(kvals[1]) ** 2 * (st + ds * k1 / 2) ** 4)
    if temp < 0.0:
        return None
    k2 = ds / (st + ds * k1 / 2) * math.sqrt(temp)

    temp = \
        (gamma ** 2 * smax ** 2 - abs(kvals[1]) ** 2 * (st + ds * k2 / 2) ** 4)
    if temp < 0.0:
        return None
    k3 = ds / (st + ds * k2 / 2) * math.sqrt(temp)

    temp = \
        (gamma ** 2 * smax ** 2 - abs(kvals[2]) ** 2 * (st + ds * k3) ** 4)
    if temp < 0.0:
        return None
    k4 = ds / (st + ds * k3) * math.sqrt(temp)

    return k1 / 6 + k2 / 3 + k3 / 3 + k4 / 6


#  Arc length code translated from matlab
#    (c) Michael Lustig 2005
#    modified 2006 and 2007
#    Rewritten in Python in 2020 by Kevin Johnson
def min_time_gradient(c: np.ndarray, g0=0, gfin=0, gmax=4, smax=15,
                      dt=4e-3, gamma=4.257):
    r"""
    Given a k-space trajectory c(n), gradient and slew constraints. This
    function will return a new parametrization that will meet these
    constraint while getting from one point to the other in minimum time.

    Args:
        c (array): Curve in k-space given in any parametrization [1/cm]
                        Nx3 real array
        g0 (float): Initial gradient amplitude (leave empty for g0 = 0)
        gfin (float): Gradient value at the end of the trajectory. If not
                        possible, the result would be the largest possible
                        ampltude. (Leave empty if you don't care to get
                        maximum gradient.)
        gmax (float): Maximum gradient [G/cm] (3.9 default)
        smax (float): Maximum slew [G/Cm/ms]  (14.5 default)
        dt (float): Sampling time interval [ms] (4e-3 default)
        gamma (float): Gyromagnetic ratio

    Returns:
        tuple: (g, k, s, t) tuple containing

        - **g** - (array): gradient waveform [G/cm]
        - **k** - (array): exact k-space corresponding to gradient g.
        - **s** - (array): slew rate [G/cm/ms]
        - **time** - (array):  sampled time

    References:
        Lustig M, Kim SJ, Pauly JM. A fast method for designing time-optimal
        gradient waveforms for arbitrary k-space trajectories. IEEE Trans Med
        Imaging. 2008;27(6):866-873. doi:10.1109/TMI.2008.922699
    """

    def sdotmax(cs: interpolate.CubicSpline, s: np.ndarray,
                gmax, smax, gamma=4.257):
        # [sdot, k, ] = sdotMax(PP, p_of_s, s, gmax, smax)
        #
        # Given a k-space curve C (in [1/cm] units), maximum gradient amplitude
        # (in G/cm) and maximum slew-rate (in G/(cm*ms)).
        # This function calculates the upper bound for the time parametrization
        # sdot (which is a non scaled max gradient constaint) as a function
        # of s.
        #
        #   cs      --  spline polynomial
        #   p_of_s  --  parametrization vs arclength
        #   s       --  arclength parametrization (0->1)
        #   gmax    --  maximum gradient (G/cm)
        #   smax    --  maximum slew rate (G/ cm*ms)
        #
        #   returns the maximum sdot (1st derivative of s) as a function of
        #   arclength s
        #   Also, returns curvature as a function of s and length of curve (L)
        #
        #  (c) Michael Lustig 2005
        #  last modified 2006

        # Absolute value of 2nd derivative in curve space using cubic splines
        cs2 = cs.derivative(2)  # spline derivative
        cs2_highres = cs2(s)  # evaluated along arc length
        k = np.linalg.norm(cs2_highres, axis=1)  # magnitude

        # calc I constraint curve (maximum gradient)
        sdot1 = gamma * gmax * np.ones_like(s)

        # calc II constraint curve (curve curvature dependent)
        sdot2 = np.sqrt(gamma * smax / (k + np.finfo(float).eps))

        # calc total constraint
        sdot = np.minimum(sdot1, sdot2)

        return sdot, k

    # Curve in arbitrary paramater space, cubic spline
    num_p = c.shape[0]
    p = np.linspace(0, 1, num_p, endpoint=True)
    cp = interpolate.CubicSpline(p, c, axis=0)

    # Integrate absolute value to find length and s(arc) vs p(paramater)
    cp1_spline = cp.derivative()
    p_highres = np.linspace(0, 1, num_p * 10)
    cp1_highres = cp1_spline(p_highres)
    ds_p = np.linalg.norm(cp1_highres, axis=1)

    # s vs p to enable conversion
    s_of_p = integrate.cumtrapz(ds_p, p_highres, initial=0)
    curve_length = s_of_p[-1]

    # decide ds and compute st for the first point
    stt0 = (gamma * smax)  # always assumes first point is max slew
    st0 = stt0 * dt / 8  # start at 1/8 the gradient for accuracy close to g=0
    s0 = st0 * dt
    ds = s0 / 4.0  # smaller step size for numerical accuracy
    ns = int(curve_length / ds)

    if g0 is None:
        g0 = 0

    # s is arc length at high resolution
    s = np.linspace(0, curve_length, ns, endpoint=True)

    # Cubic spline at s positions (s of p)
    cp_highres = cp(p_highres)
    cs = interpolate.CubicSpline(s_of_p, cp_highres, axis=0)

    # compute constraints (forbidden line curve)
    phi, k = sdotmax(cs, s, gmax, smax)

    # extend for the Runge-Kutte method
    k = np.pad(k, (0, 3), 'constant', constant_values=(0,))

    # Get the start
    sta = np.zeros_like(s)
    sta[0] = min(g0 * gamma + st0, gamma * gmax)

    # solve ODE forward
    for n in range(1, s.shape[0]):
        kpos = n
        dstds = runge_kutta(ds, sta[n - 1], k[kpos:kpos + 4], smax)

        if dstds is None:
            sta[n] = phi[n]
        else:
            tmpst = sta[n - 1] + dstds
            sta[n] = min(tmpst, phi[n])

    stb = 0 * s
    if gfin is None:
        stb[-1] = sta[-1]
    else:
        stb[-1] = min(max(gfin * gamma, st0), gamma * gmax)

    # solve ODE backwards
    for n in range(s.shape[0] - 2, 0, -1):

        kpos_end = n  # to 0
        kpos = kpos_end + 3
        dstds = runge_kutta(ds, stb[n + 1], k[kpos:(kpos - 3):-1], smax)

        if dstds is None:
            stb[n] = phi[n - 1]
        else:
            tmpst = stb[n + 1] + dstds
            stb[n] = min(tmpst, phi[n - 1])

    # Fix last point which is indexed a bit off
    n = 0
    kpos_end = n
    kpos = kpos_end + 3
    dstds = runge_kutta(ds, stb[n + 1], k[kpos::-1], smax)
    if dstds is None:
        stb[n] = phi[n * 2 - 1]
    else:
        tmpst = stb[n + 1] + dstds
        stb[n] = min(tmpst, phi[n - 1])

    # take the minimum of the curves
    ds = s[1] - s[0]
    st_of_s = np.minimum(sta, stb)

    # compute time
    t_of_s = integrate.cumtrapz(1. / st_of_s, initial=0) * ds

    t = np.arange(0, t_of_s[-1] + np.finfo(float).eps, dt)

    t_of_s = interpolate.CubicSpline(t_of_s, s)
    s_of_t = t_of_s(t)
    c = np.squeeze(cs(s_of_t))

    g = np.diff(c, axis=0, append=np.zeros((1, 3))) / gamma / dt
    g[-1, :] = g[-2, :] + g[-2, :] - g[-3, :]

    k = integrate.cumtrapz(g, t, initial=0, axis=0) * gamma

    s = np.diff(g, axis=0) / dt

    return g, k, s, t
