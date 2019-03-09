import sigpy as sp


__all__ = ['espirit_maps']


def espirit_maps(ksp, calib_width=24,
                 thresh=0.001, kernel_width=6,
                 crop=0.8,
                 max_power_iter=30, device=sp.cpu_device,
                 output_eigenvalue=False):
    """Generate ESPIRiT maps from k-space.

    Currently only supports outputting one set of maps.

    Args:
        ksp (array): k-space array of shape [num_coils, n_ndim, ..., n_1]
        calib (tuple of ints): length-2 image shape.
        thresh (float): threshold for the calibration matrix.
        kernel_width (int): kernel width for the calibration matrix.
        max_power_iter (int): maximum number of power iterations.
        device (Device): computing device.
        crop (int): cropping threshold.

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
    with sp.get_device(ksp):
        # Get calibration region
        calib_shape = [num_coils] + [calib_width] * img_ndim
        calib = sp.resize(ksp, calib_shape)
        calib = sp.to_device(calib, device)

    device = sp.Device(device)
    xp = device.xp
    with device:
        # Get calibration matrix
        kernel_shape = [num_coils] + [kernel_width] * img_ndim
        kernel_strides = [1] * (img_ndim + 1)
        mat = sp.array_to_blocks(calib, kernel_shape, kernel_strides)
        mat = mat.reshape([-1, sp.prod(kernel_shape)])

        # Perform SVD on calibration matrix
        _, S, VH = xp.linalg.svd(mat, full_matrices=False)
        VH = VH[S > thresh * S.max(), :]

        # Get kernels
        num_kernels = len(VH)
        kernels = VH.reshape([num_kernels] + kernel_shape)
        img_shape = ksp.shape[1:]

        # Get covariance matrix in image domain
        AHA = xp.zeros(img_shape[::-1] + (num_coils, num_coils),
                       dtype=ksp.dtype)
        for kernel in kernels:
            img_kernel = sp.ifft(sp.resize(kernel, ksp.shape),
                                 axes=range(-img_ndim, 0))
            aH = xp.expand_dims(img_kernel.T, axis=-1)
            a = xp.conj(aH.swapaxes(-1, -2))
            AHA += aH @ a

        AHA *= (sp.prod(img_shape) / kernel_width**img_ndim)

        # Power Iteration to compute top eigenvector
        mps = xp.ones(ksp.shape[::-1] + (1, ), dtype=ksp.dtype)
        for _ in range(max_power_iter):
            sp.copyto(mps, AHA @ mps)
            eig_value = xp.sum(xp.abs(mps)**2, axis=-2, keepdims=True)**0.5
            mps /= eig_value

        # Normalize phase with respect to first channel
        mps = mps.T[0]
        mps *= xp.conj(mps[0] / xp.abs(mps[0]))

        # Crop maps by thresholding eigenvalue
        eig_value = eig_value.T[0]
        mps *= eig_value > crop

        if output_eigenvalue:
            return mps, eig_value
        else:
            return mps
