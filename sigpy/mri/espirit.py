import sigpy as sp


__all__ = ['espirit_calib']


def espirit_calib(ksp, thresh=0.001, kernel_width=12, max_power_iter=30):
    """Generate ESPIRiT maps from k-space.

    Currently only supports outputting one set of maps.

    Args:
        ksp (array): k-space array of shape [num_coils] + calib_shape.

    Returns:
        array: ESPIRiT maps of the same shape as ksp.

    References:
        Martin Uecker, Peng Lai, Mark J. Murphy, Patrick Virtue, Michael Elad,
        John M. Pauly, Shreyas S. Vasanawala, and Michael Lustig
        ESPIRIT - An Eigenvalue Approach to Autocalibrating Parallel MRI:
        Where SENSE meets GRAPPA.
        Magnetic Resonance in Medicine, 71:990-1001 (2014)

    """
    img_ndim = ksp.ndim - 1
    num_coils = len(ksp)
    device = sp.get_device(ksp)
    xp = device.xp
    with device:
        # Get calibration matrix
        kernel_shape = [num_coils] + [kernel_width] * img_ndim
        kernel_strides = [1] * (img_ndim + 1)
        mat = sp.array_to_blocks(ksp, kernel_shape, kernel_strides)
        mat = mat.reshape([-1, sp.prod(kernel_shape)])
        
        # Perform SVD on calibration matrix
        _, S, VH = xp.linalg.svd(mat, full_matrices=False)
        VH = VH[S > thresh * S.max(), :]

        # Get kernels
        num_kernels = len(VH)
        kernels = VH.reshape([num_kernels] + kernel_shape)
        img_kernels = sp.ifft(sp.resize(kernels, (num_kernels, ) + ksp.shape),
                              axes=[-1, -2])

        # Eigenvalue decomposition
        AH = img_kernels.T.reshape(ksp.shape[::-1] + (num_kernels, ))
        AHA = AH @ AH.swapaxes(-1, -2).conjugate()
        mps = sp.randn(ksp.shape[::-1] + (1, ), dtype=ksp.dtype, device=device)
        for _ in range(max_power_iter):
            sp.copyto(mps, AHA @ mps)
            eig_value = xp.sum(xp.abs(mps)**2, axis=-2, keepdims=True)**0.5
            mps /= eig_value

        # Normalize phase with respect to first channel
        mps = mps.T[0]
        mps *= xp.conj(mps[0] / xp.abs(mps[0]))

        return mps
