# -*- coding: utf-8 -*-
"""Methods for Echo-Planar Imaging (EPI) acquisition:

* retrospectively undersampling phase-encoding direction

Author:
    Zhengguo Tan <zhengguo.tan@gmail.com>
"""
import numpy as np


def unsamp_ky(kdat, phaenc_axis=-3,
              uniform_unsamp=True, unsamp_factor=2):
    # find valid phase-encoding lines
    kdat1 = np.swapaxes(kdat, phaenc_axis, 0)
    kdat2 = np.reshape(kdat1, (kdat1.shape[0], -1))
    kdat3 = np.sum(kdat2, axis=1)

    sampled_phaenc_ind = np.array(np.nonzero(kdat3)).ravel()
    sampled_phaenc_len = len(sampled_phaenc_ind)

    loop_shape = [np.prod(kdat.shape[:phaenc_axis])] \
        + list(kdat.shape[phaenc_axis:])
    kdat4 = np.reshape(kdat, loop_shape)

    output = np.zeros_like(kdat4)

    shift_cnt = 0

    # loop over all high dimensions
    for l in range(loop_shape[0]):
        if uniform_unsamp:
            rinds = np.arange(shift_cnt, sampled_phaenc_len, unsamp_factor)
            shift_cnt = (shift_cnt + 1) % unsamp_factor
        else:
            rinds = np.random.randint(sampled_phaenc_len,
                                      size=(sampled_phaenc_len
                                            // unsamp_factor))

        rand_unsamp_lines = sampled_phaenc_ind[rinds]

        tmp = kdat4[l, rand_unsamp_lines, ...]
        output[l, rand_unsamp_lines, ...] = tmp

    return np.reshape(output, kdat.shape)


def split_shots(kdat, phaenc_axis=-2, shots=2):
    """split shots within one diffusion encoding
    """
    # find valid phase-encoding lines
    kdat1 = np.swapaxes(kdat, phaenc_axis, 0)
    kdat2 = np.reshape(kdat1, (kdat1.shape[0], -1))

    kdat3 = np.sum(kdat2, axis=1)
    sampled_phaenc_ind = np.array(np.nonzero(kdat3)).ravel()
    sampled_phaenc_len = len(sampled_phaenc_ind)

    out_shape = [shots] + list(kdat2.shape)
    output = np.zeros_like(kdat2, shape=out_shape)

    for l in range(sampled_phaenc_len):
        s = l % shots

        ind = sampled_phaenc_ind[l]
        output[s, ind, :] = kdat2[ind, :]

    output = np.reshape(output, [shots] + list(kdat1.shape))
    output = np.swapaxes(output, 1, phaenc_axis)

    return output
