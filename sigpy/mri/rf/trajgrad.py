"""MRI gradient and excitation trajectory design
"""

import numpy as np

__all__ = ['mintrapgrad', 'trapgrad', 'spiralvarden', 'spiralarch']


def mintrapgrad(area, gmax, dgdt, dt, *args):
    """Minimal duration trapezoidal gradient designer.

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
        # we get the solution for plateau amp by setting derivative of
        # duration as a function of amplitude to zero and solving
        a = np.sqrt(dgdt * area / 2)

        # finish design with discretization
        # make a flat portion of magnitude a
        # and enough area for the entire swath
        print(area / gmax / dt)
        flat = np.ones(1, np.floor(area / a / dt))
        flat = flat / np.sum(flat) * area / dt
        if max(flat) > gmax:
            flat = np.ones(1, np.ceil(area / gmax / dt))
            flat = flat / sum(flat) * area / dt

        # make attack and decay ramps

        ramppts = int(np.ceil(max(flat) / dgdt / dt))
        trap = np.concatenate((np.linspace(0, ramppts, num=ramppts) / ramppts * np.max(flat), flat,
                              np.linspace(ramppts, 0, num=ramppts) / ramppts * np.max(flat)))

    else:
        trap, ramppts = 0, 0

    return trap, ramppts


def trapgrad(area, gmax, dgdt, dt, *args):
    """General trapezoidal gradient designer. Min total time.

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
                pulse = np.concatenate((np.linspace(0, ramppts, num=ramppts)/ramppts, np.linspace(ramppts, 0, num=ramppts)/ramppts))
            else:
                # trapezoid pulse
                nflat = np.ceil((area - triareamax)/gmax / dt / 2) * 2
                pulse = np.concatenate((np.linspace(0, ramppts, num=ramppts) / ramppts, np.ones(int(nflat)), np.linspace(ramppts, 0, num=ramppts) / ramppts))

            trap = pulse * (area / (sum(pulse) * dt))

        else:
            # make a flat portion of magnitude gmax
            # and enough area for the entire swath
            flat = np.ones(1, np.ceil(area/gmax/dt))
            flat = flat / sum(flat) * area / dt

            # make attack and decay ramps
            ramppts = np.ceil(np.max(flat) / dgdt / dt)
            trap = np.concatenate(np.linspace(0, ramppts, num=ramppts) / ramppts * np.max(flat), flat, np.linspace(ramppts, 0, num=ramppts) / ramppts * np.max(flat))

    else:
        trap, ramppts = 0, 0

    return trap, ramppts


def spiralvarden(opfov, opxres, gts, gslew, gamp, densamp, dentrans, nl):
    """Variable density spiral designer. Produces trajectory, gradients,
    and slew rate.

    Args:
        opfov (float): imaging field of view (cm)
        opxres (float): imaging resolution (cm)
        gts (float): sample time in sec
        gslew (float): max slew rate in T/m/s
        gamp (float): max gradient amplitude in T/m/s
        densamp (float):  duration of full density sampling (# of samples)
        dentrans (float): duration of transition from higher to lower (should be >= densamp/2)
        nl (float): degree of undersampling outer region

    """
    fsgcm = gamp  # fullscale g/cm
    risetime = gamp / gslew * 10000  # us
    ts = gts  # sampling time
    gts = gts  # gradient sampling time
    targetk = opxres / 2
    A = 32766  # output scaling of waveform (fullscale)

    MAXDECRATIO = 32
    GAM = 4257.0
    S = (gts / 1e-6) * A / risetime
    dr = ts / gts
    OMF = 2.0 * np.pi * opfov / (1 / (GAM * fsgcm * gts))
    OM = 2.0 * np.pi / nl * opfov / (1 / (GAM * fsgcm * gts))
    distance = 1.0 / (opfov * GAM * fsgcm * gts / A)

    ac = A
    loop = 1
    absk = 0
    decratio = 1
    S0 = gslew * 100
    ggx, ggy = [], []
    dens = []
    kx, ky = [], []

    while loop > 0:
        loop = 0
        om = OM / decratio
        omf = OMF / decratio
        s = S / decratio
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
            realn = n / decratio
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
                    scthat = scoffset + (omf + (om - omf) * fractrans) * (tauhat - denoffset)

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
                    decratio = decratio * 2.0

                    if decratio > MAXDECRATIO:
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

            if np.remainder(n, decratio) == 0:
                m = int(np.round(n / decratio))
                gx = np.round((tkx - oldkx) / decratio)
                gx = gx - np.remainder(gx, 2)
                gy = np.round((tky - oldky) / decratio)
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
                    m = int(m / dr)  # JBM added int
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
    delk = 1 / 4.258 / opfov  # (g ms)/cm

    # ramp down
    l2 = len(g) - 1
    rsteps = np.ceil(np.abs(g[l2]) / (S0 * 0.99) / gts)
    ind3 = l2 + np.linspace(1, rsteps, num=rsteps)
    c = g[l2] * np.linspace(rsteps, 0, num=rsteps) / rsteps
    g.extend(c)
    dens.extend([0] * len(ind3))

    # rewinder
    rewx, ramppts = trapgrad(abs(np.real(sum(g))) * gts, gamp, gslew * 50, gts)
    rewy, ramppts = trapgrad(abs(np.imag(sum(g))) * gts, gamp, gslew * 50, gts)

    # append rewinder gradient
    if len(rewx) > len(rewy):
        r = -np.sign(np.real(sum(g))) * rewx - np.sign(np.imag(sum(g))) * 1j * np.abs(np.imag(sum(g))) / np.real(
            sum(g)) * rewx
    else:
        r = -np.sign(np.real(sum(g))) * np.abs(np.real(sum(g)) / np.imag(sum(g))) * rewy - 1j * np.sign(
            np.imag(sum(g))) * rewy

    g = np.concatenate((g, r))

    # change from (real, imag) notation to (Nt, 2) notation
    gtemp = np.zeros((len(g), 2))
    gtemp[:, 0] = np.real(g)
    gtemp[:, 1] = np.imag(g)
    g = gtemp

    # calculate trajectory, slew rate factor from designed gradient
    k = np.cumsum(g, axis=0) * dt / delk / opfov  # trajectory
    t = np.linspace(0, len(g), num=len(g) + 1)  # time vector
    s = np.diff(g, axis=0) / (gts * 1000)  # slew rate factor

    return g, k, t, s, dens


def spiralarch(D, N, gts, gslew, gamp):
    """Analytic archimedean spiral designer. Produces trajectory, gradients,
    and slew rate.

    Args:
        D (float): imaging field of view (m)
        N(float): effective matrix size
        gts (float): sample time in sec
        gslew (float): max slew rate in T/m/s
        gamp (float): max gradient amplitude in T/m

    References:
        Glover, G. H.(1999).
        Simple Analytic Spiral K-Space Algorithm.
        Magnetic resonance in medicine, 42, 412-415.

        Bernstein, M.A.; King, K.F.; amd Zhou, X.J. (2004).
        Handbook of MRI Pulse Sequences. Elsevier.
    """

    gam = 267.522 * 1e6  # rad/s/Tesla
    gambar = gam / 2 / np.pi  # Hz/T
    dx = D / N  # m, resolution
    lam = 1 / (2 * np.pi * D)
    beta = gambar * gslew / lam

    kmax = N / (2 * D)
    dr = 1 / (2 * kmax)
    a_2 = (9 * beta / 4) ** (1 / 3)  # rad ** (1/3) / s ** (2/3)
    Lambda = 5
    thetamax = kmax / lam
    Ts = (3 * gam * gamp / (4 * np.pi * lam * a_2 ** 2)) ** 3
    theta_s = (0.5 * beta * Ts ** 2) / (Lambda + beta / (2 * a_2) * Ts ** (4 / 3))
    t_g = np.pi * lam * (thetamax ** 2 - theta_s ** 2) / (gam * gamp)
    n_s = int(np.round(Ts / gts))
    n_g = int(np.round(t_g / gts))

    if thetamax > theta_s:
        print(' Spiral trajectory is slewrate limited or amplitude limited')

        n_t = n_s + n_g
        tacq = Ts + t_g

        t_s = np.linspace(0, Ts, n_s)
        t_g = np.linspace(Ts + gts, tacq, n_g)

        theta_1 = (beta / 2 * t_s ** 2) / (Lambda + beta / (2 * a_2) * t_s ** (4 / 3))
        dtheta_1 = -(2 / 3) * a_2 * beta * t_s * (beta * t_s ** (4 / 3) - 6 * a_2 * Lambda) / (
                    beta * t_s ** (4 / 3) + 6 * a_2 * Lambda) ** 2  # quotient rule
        theta_2 = np.sqrt(theta_s ** 2 + gam / (np.pi * lam) * gamp * (t_g - Ts))
        dtheta_2 = gslew * gam / (
                    2 * lam * np.sqrt(np.pi / lam * (gslew * gam * t_g - gslew * gam * Ts + lam * np.pi * theta_s)))

        k1 = lam * theta_1 * (np.cos(theta_1) + 1j * np.sin(theta_1))
        k2 = lam * theta_2 * (np.cos(theta_2) + 1j * np.sin(theta_2))
        k = np.concatenate((k1, k2), axis=0)

    else:

        tacq = 2 * np.pi * D / 3 * np.sqrt(np.pi / (gam * gslew * dx ** 3))
        n_t = int(np.round(tacq / gts))
        t_s = np.linspace(0, tacq, n_t)
        theta_1 = (beta / 2 * t_s ** 2) / (Lambda + beta / (2 * a_2) * t_s ** (4 / 3))

        k = lam * theta_1 * (np.cos(theta_1) + 1j * np.sin(theta_1))

    # end of trajectory calculation; prepare outputs
    g = np.diff(k, 1, axis=0) / (gts * gambar)  # gradient
    s = np.diff(g, 1, axis=0) / (gts * 1000)  # slew rate factor

    # change from (real, imag) notation to (Nt, 2) notation
    ktemp = np.zeros((len(k), 2))
    ktemp[:, 0], ktemp[:, 1] = np.real(k), np.imag(k)
    k = ktemp

    gtemp = np.zeros((len(g), 2))
    gtemp[:, 0], gtemp[:, 1] = np.real(g), np.imag(g)
    g = gtemp

    stemp = np.zeros((len(s), 2))
    stemp[:, 0], stemp[:, 1] = np.real(s), np.imag(s)
    s = stemp

    t = np.linspace(0, len(g), num=len(g) + 1)  # time vector

    return g, k, t, s