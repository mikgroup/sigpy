"""MRI gradient and excitation trajectory design
"""

import numpy as np

__all__ = ['min_trap_grad', 'trap_grad', 'spiral_varden', 'spiral_arch',
           'stack_of', 'traj_array2complex', 'traj_complex2array']


def min_trap_grad(area, gmax, dgdt, dt):
    r"""Minimal duration trapezoidal gradient designer.

    Args:
        area (float): pulse area in (g*sec)/cm
        gmax (float): maximum gradient in g/cm
        dgdt (float): max slew rate in g/cm/sec
        dt (float): sample time in sec

    """

    if np.abs(area) > 0:
        # we get the solution for plateau amp by setting derivative of
        # duration as a function of amplitude to zero and solving
        a = np.sqrt(dgdt * area / 2)

        # finish design with discretization
        # make a flat portion of magnitude a and enough area for the swath
        flat = np.ones(1, np.floor(area / a / dt))
        flat = flat / np.sum(flat) * area / dt
        if max(flat) > gmax:
            flat = np.ones(1, np.ceil(area / gmax / dt))
            flat = flat / sum(flat) * area / dt

        # make attack and decay ramps
        ramppts = int(np.ceil(max(flat) / dgdt / dt))
        ramp_up = np.linspace(0, ramppts, num=ramppts) / ramppts * np.max(flat)
        ramp_dn = np.linspace(ramppts, 0, num=ramppts) / ramppts * np.max(flat)

        trap = np.concatenate((ramp_up, flat, ramp_dn))

    else:
        # negative-area trap requested?
        trap, ramppts = 0, 0

    return trap, ramppts


def trap_grad(area, gmax, dgdt, dt, *args):
    r"""General trapezoidal gradient designer. Min total time.

    Args:
        area (float): pulse area in (g*sec)/cm
        gmax (float): maximum gradient in g/cm
        dgdt (float): max slew rate in g/cm/sec
        dt (float): sample time in sec

    """

    if len(args) < 5:
        # in case we are making a rewinder
        rampsamp = 1

    if np.abs(area) > 0:
        if rampsamp:

            ramppts = np.ceil(gmax/dgdt/dt)
            triareamax = ramppts * dt * gmax

            if triareamax > np.abs(area):
                # triangle pulse
                newgmax = np.sqrt(np.abs(area) * dgdt)
                ramppts = np.ceil(newgmax/dgdt/dt)
                ramp_up = np.linspace(0, ramppts, num=ramppts)/ramppts
                ramp_dn = np.linspace(ramppts, 0, num=ramppts)/ramppts
                pulse = np.concatenate((ramp_up, ramp_dn))
            else:
                # trapezoid pulse
                nflat = np.ceil((area - triareamax)/gmax / dt / 2) * 2
                ramp_up = np.linspace(0, ramppts, num=ramppts) / ramppts
                ramp_dn = np.linspace(ramppts, 0, num=ramppts) / ramppts
                pulse = np.concatenate((ramp_up, np.ones(int(nflat)), ramp_dn))

            trap = pulse * (area / (sum(pulse) * dt))

        else:
            # make a flat portion of magnitude gmax
            # and enough area for the entire swath
            flat = np.ones(1, np.ceil(area/gmax/dt))
            flat = flat / sum(flat) * area / dt
            flat_top = np.max(flat)

            # make attack and decay ramps
            ramppts = np.ceil(np.max(flat) / dgdt / dt)
            ramp_up = np.linspace(0, ramppts, num=ramppts) / ramppts * flat_top
            ramp_dn = np.linspace(ramppts, 0, num=ramppts) / ramppts * flat_top
            trap = np.concatenate((ramp_up, flat, ramp_dn))

    else:
        trap, ramppts = 0, 0

    return trap, ramppts


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
        rewx, ramppts = trap_grad(abs(np.real(sum(g))) * gts,
                                  gamp, gslew * 50, gts)
        rewy, ramppts = trap_grad(abs(np.imag(sum(g))) * gts,
                                  gamp, gslew * 50, gts)

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
    Ts = (3 * gam * gamp / (4 * np.pi * lam * a_2 ** 2)) ** 3
    theta_s = 0.5 * beta * Ts ** 2
    theta_s /= (lamb + beta / (2 * a_2) * Ts ** (4 / 3))
    t_g = np.pi * lam * (theta_max ** 2 - theta_s ** 2) / (gam * gamp)
    n_s = int(np.round(Ts / gts))
    n_g = int(np.round(t_g / gts))

    if theta_max > theta_s:
        print(' Spiral trajectory is slewrate limited or amplitude limited')

        n_t = n_s + n_g
        tacq = Ts + t_g

        t_s = np.linspace(0, Ts, n_s)
        t_g = np.linspace(Ts + gts, tacq, n_g)

        theta_1 = beta / 2 * t_s ** 2
        theta_1 /= (lamb + beta / (2 * a_2) * t_s ** (4 / 3))
        theta_2 = theta_s ** 2 + gam / (np.pi * lam) * gamp * (t_g - Ts)
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
    k = traj_complex2array(k)
    g = traj_complex2array(g)
    s = traj_complex2array(s)

    t = np.linspace(0, len(g), num=len(g) + 1)  # time vector

    return g, k, t, s


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
    k = traj_array2complex(k)

    for ii in range(num):
        kr = k[0:] * np.exp(2 * np.pi * 1j * ii / num)
        z_coord = np.expand_dims(np.ones(len(kr)) * z[ii], axis=1)
        krz = np.concatenate((traj_complex2array(kr), z_coord), axis=1)

        kout[ii*len(krz):(ii+1)*len(krz), :] = krz

    return kout


def traj_complex2array(k):
    r"""Function to convert complex convention trajectory to [Nt 2] trajectory

    Args:
        k (complex array): Nt vector
    """
    kout = np.zeros((len(k), 2))
    kout[:, 0], kout[:, 1] = np.real(k), np.imag(k)
    return kout


def traj_array2complex(k):
    r"""Function to convert [Nt Nd] convention traj to complex convention

    Args:
        k (complex array): Nt vector
    """
    kout = k[:, 0] + 1j * k[:, 1]
    return kout
