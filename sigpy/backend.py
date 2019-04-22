# -*- coding: utf-8 -*-
"""Functions and classes for getting and setting computing devices.

"""
import numpy as np
from sigpy import config
if config.cupy_enabled:
    import cupy as cp

if config.mpi4py_enabled:
    from mpi4py import MPI

    if config.nccl_enabled:
        from cupy.cuda import nccl


__all__ = ['Device', 'get_device', 'get_array_module', 'cpu_device',
           'to_device', 'copyto', 'Communicator']


class Device(object):
    """Device class.

    This class extends cupy.Device, with id > 0 representing the id_th GPU,
    and id = -1 representing CPU. cupy must be installed to use GPUs.

    The array module for the corresponding device can be obtained via .xp.
    Similar to cupy.Device, the Device object can be used as a context:

        >>> device = Device(2)
        >>> xp = device.xp  # xp is cupy.
        >>> with device:
        >>>     x = xp.array([1, 2, 3])
        >>>     x += 1

    Args:
        id_or_device (int or Device or cupy.cuda.Device): id > 0 represents
            the corresponding GPUs, and id = -1 represents CPU.

    Attributes:
        id (int): id = -1 represents CPU,
            and others represents the id_th GPUs.

    """

    def __init__(self, id_or_device):
        if isinstance(id_or_device, int):
            id = id_or_device
        elif isinstance(id_or_device, Device):
            id = id_or_device.id
        elif config.cupy_enabled and isinstance(id_or_device, cp.cuda.Device):
            id = id_or_device.id
        else:
            raise ValueError(
                'Accepts int, Device or cupy.cuda.Device, got {}'.format(
                    id_or_device))

        if id != -1:
            if config.cupy_enabled:
                self.cpdevice = cp.cuda.Device(id)
            else:
                raise ValueError(
                    'cupy not installed, but set device {}.'.format(id))

        self.id = id

    @property
    def xp(self):
        """module: numpy or cupy module for the device."""
        if self.id == -1:
            return np

        return cp

    def __int__(self):
        return self.id

    def __eq__(self, other):
        if isinstance(other, int):
            return self.id == other
        elif isinstance(other, Device):
            return self.id == other.id
        elif config.cupy_enabled and isinstance(other, cp.cuda.Device):
            return self.id == other.id
        else:
            return False

    def __ne__(self, other):
        return not self == other

    def __enter__(self):
        if self.id == -1:
            return None

        return self.cpdevice.__enter__()

    def __exit__(self, *args):
        if self.id == -1:
            pass
        else:
            self.cpdevice.__exit__()

    def __repr__(self):
        if self.id == -1:
            return '<CPU Device>'

        return self.cpdevice.__repr__()


cpu_device = Device(-1)


def get_array_module(array):
    """Gets an appropriate module from :mod:`numpy` or :mod:`cupy`.

    This is almost equivalent to :func:`cupy.get_array_module`. The differences
    are that this function can be used even if cupy is not available.

    Args:
        array: Input array.

    Returns:
        module: :mod:`cupy` or :mod:`numpy` is returned based on input.
    """
    if config.cupy_enabled:
        return cp.get_array_module(array)
    else:
        return np


def get_device(array):
    """Get Device from input array.

    Args:
        array (array): Array.

    Returns:
        Device.

    """
    if get_array_module(array) == np:
        return cpu_device
    else:
        return Device(array.device)


def to_device(input, device=cpu_device):
    """Move input to device. Does not copy if same device.

    Args:
        input (array): Input.
        device (int or Device or cupy.Device): Output device.

    Returns:
        array: Output array placed in device.
    """
    idevice = get_device(input)
    odevice = Device(device)

    if idevice == odevice:
        return input

    if odevice == cpu_device:
        with idevice:
            return input.get()
    else:
        with odevice:
            return cp.asarray(input)


def copyto(output, input):
    """Copy from input to output. Input/output can be in different device.

    Args:
        input (array): Input.
        output (array): Output.

    """
    idevice = get_device(input)
    odevice = get_device(output)
    if idevice == cpu_device and odevice != cpu_device:
        with odevice:
            output.set(input)
    elif idevice != cpu_device and odevice == cpu_device:
        with idevice:
            np.copyto(output, input.get())
    else:
        idevice.xp.copyto(output, input)


class Communicator(object):
    """Communicator for distributed computing using MPI.

    When NCCL is not installed, arrays are moved to CPU,
    then communicated through MPI, and moved back
    to original device.
    When mpi4py is not installed, the communicator errors.

    """

    def __init__(self):
        if config.mpi4py_enabled:
            self.mpi_comm = MPI.COMM_WORLD
            self.size = self.mpi_comm.Get_size()
            self.rank = self.mpi_comm.Get_rank()
        else:
            self.size = 1
            self.rank = 0

        # Keep nccl comms for reuse
        if config.nccl_enabled:
            self.nccl_comms = {}

    def allreduce(self, input):
        """All reduce operation in-place.

        Sums input across all nodes and broadcast back to each node.

        Args:
            input (array): input array.

        """
        if self.size > 1:
            if config.nccl_enabled:
                device = get_device(input)
                devices = self.mpi_comm.allgather(device.id)
                if all([d >= 0 for d in devices]):
                    nccl_comm = self._get_nccl_comm(device, devices)
                    nccl_dtype, nccl_size = self._get_nccl_dtype_size(input)
                    with device:
                        nccl_comm.allReduce(input.data.ptr,
                                            input.data.ptr,
                                            nccl_size, nccl_dtype,
                                            nccl.NCCL_SUM,
                                            cp.cuda.Stream.null.ptr)
                        return

            cpu_input = to_device(input, cpu_device)
            self.mpi_comm.Allreduce(MPI.IN_PLACE, cpu_input)
            copyto(input, cpu_input)

    def reduce(self, input, root=0):
        """Reduce operation in-place.

        Sums input across all nodes in root node.

        Args:
            input (array): input array.
            root (int): root node rank.

        """
        if self.size > 1:
            cpu_input = to_device(input, cpu_device)
            if self.rank == root:
                self.mpi_comm.Reduce(MPI.IN_PLACE, cpu_input, root=root)
                copyto(input, cpu_input)
            else:
                self.mpi_comm.Reduce(cpu_input, None, root=root)

    def bcast(self, input, root=0):
        """Broadcast from root to other nodes.

        Args:
            input (array): input array.
            root (int): root node rank.

        """
        if self.size > 1:
            cpu_input = to_device(input, cpu_device)
            self.mpi_comm.Bcast(cpu_input, root=root)
            copyto(input, cpu_input)

    def gatherv(self, input, root=0):
        """Gather with variable sizes operation.

        Gather inputs across all nodes to the root node,
        and vectorizes them.

        Args:
            input (array): input array.
            root (int): root node rank.

        Returns:
            array or None: vectorized array if rank==root else None.

        """
        if self.size > 1:
            cpu_input = to_device(input, cpu_device)

            sizes = self.mpi_comm.gather(input.size, root=root)
            if self.rank == root:
                cpu_output = np.empty(sum(sizes), dtype=input.dtype)
                self.mpi_comm.Gatherv(
                    cpu_input, [cpu_output, sizes], root=root)
                return to_device(cpu_output, get_device(input))
            else:
                self.mpi_comm.Gatherv(cpu_input, [None, sizes], root=root)
        else:
            return input

    def _get_nccl_comm(self, device, devices):
        if str(devices) in self.nccl_comms:
            return self.nccl_comms[str(devices)]

        if self.rank == 0:
            nccl_comm_id = nccl.get_unique_id()
        else:
            nccl_comm_id = None

        nccl_comm_id = self.mpi_comm.bcast(nccl_comm_id)

        with device:
            nccl_comm = nccl.NcclCommunicator(
                self.size, nccl_comm_id, self.rank)
            self.nccl_comms[str(devices)] = nccl_comm

        return nccl_comm

    def _get_nccl_dtype_size(self, input):
        if input.dtype == np.float32:
            nccl_dtype = nccl.NCCL_FLOAT32
            nccl_size = input.size
        elif input.dtype == np.float64:
            nccl_dtype = nccl.NCCL_FLOAT64
            nccl_size = input.size
        elif input.dtype == np.complex64:
            nccl_dtype = nccl.NCCL_FLOAT32
            nccl_size = input.size * 2
        elif input.dtype == np.complex128:
            nccl_dtype = nccl.NCCL_FLOAT64
            nccl_size = input.size * 2
        else:
            raise ValueError(
                'dtype not supported, got {dtype}.'.format(dtype=input.dtype))

        return nccl_dtype, nccl_size
