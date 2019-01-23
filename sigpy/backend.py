import numpy as np
from sigpy import config
if config.cupy_enabled:
    import cupy as cp
    
if config.mpi4py_enabled:
    from mpi4py import MPI

    if config.nccl_enabled:
        from cupy.cuda import nccl


__all__ = ['Device', 'get_device', 'get_array_module', 'cpu_device',
           'to_device', 'copyto', 'Communicator', 'MultiGpuCommunicator']


class Device(object):
    """Device class.

    This class extends cupy.Device, with id = -1 representing CPU, and other ids representing the corresponding GPUs. 
    The array module for the corresponding device can be obtained via .xp property.
    Similar to cupy.Device, the Device object can be used as a context. For example:
       
        >>> device = Device(2)
        >>> xp = device.xp  # xp is cupy.
        >>> with device:
        >>>     x = xp.array([1, 2, 3])
        >>>     x += 1

    Args:
        id_or_device (int or Device or cupy.cuda.Device): id = -1 represents CPU.
            and other ids represents corresponding GPUs.

    Attributes:
        id (int): id = -1 represents CPU, and other ids represents corresponding GPUs.

    """
    def __init__(self, id_or_device):
        if isinstance(id_or_device, int):
            id = id_or_device
        elif isinstance(id_or_device, Device):
            id = id_or_device.id
        elif config.cupy_enabled and isinstance(id_or_device, cp.cuda.Device):
            id = id_or_device.id
        else:
            raise ValueError('Accepts int, Device or cupy.cuda.Device, got {id_or_device}'.format(
                id_or_device=id_or_device))

        if id != -1:
            if config.cupy_enabled:
                self.cpdevice = cp.cuda.Device(id)
            else:
                raise ValueError('cupy not installed, but set device {id}.'.format(id=id))

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
    """General communicator for distributed computing using MPI.

    All arrays are moved to CPU, then communicated through MPI, and moved back
    to original device.

    """
    def __init__(self):
        if config.mpi4py_enabled:
            self.mpi_comm = MPI.COMM_WORLD
            self.size = self.mpi_comm.Get_size()
            self.rank = self.mpi_comm.Get_rank()
        else:
            self.size = 1
            self.rank = 0

    def allreduce(self, input):
        if self.size == 1:
            return

        if config.mpi4py_enabled:
            mpi_buffer = to_device(input, cpu_device)
            self.mpi_comm.Allreduce(MPI.IN_PLACE, mpi_buffer)
            copyto(input, mpi_buffer)

    def reduce(self, input, root=0):
        if self.size == 1:
            return

        if config.mpi4py_enabled:
            mpi_buffer = to_device(input, cpu_device)
            if self.rank == root:
                self.mpi_comm.Reduce(MPI.IN_PLACE, mpi_buffer, op=MPI.SUM, root=root)
                copyto(input, mpi_buffer)
            else:
                self.mpi_comm.Reduce(mpi_buffer, None, op=MPI.SUM, root=root)


class MultiGpuCommunicator(Communicator):
    """Communicator for distributed computing between multiple GPUs.

    If nccl is installed with cupy, then nccl will be used. Otherwise,
    reduces to Communicator.

    """
    def __init__(self):
        if config.mpi4py_enabled:
            self.mpi_comm = MPI.COMM_WORLD
            self.size = self.mpi_comm.Get_size()
            self.rank = self.mpi_comm.Get_rank()
        else:
            self.size = 1
            self.rank = 0
            
        self.device = Device(self.rank % cp.cuda.runtime.getDeviceCount())

        if config.nccl_enabled:
            if self.rank == 0:
                nccl_comm_id = nccl.get_unique_id()
            else:
                nccl_comm_id = None

            nccl_comm_id = self.mpi_comm.bcast(nccl_comm_id)

            with self.device:
                self.nccl_comm = nccl.NcclCommunicator(self.size, nccl_comm_id, self.rank)

    if config.nccl_enabled:
        def allreduce(self, input):
            if self.device != get_device(input):
                raise ValueError('Input device is different from communicator device.')

            if self.size == 1:
                return

            nccl_dtype, nccl_size = self._get_nccl_dtype_size(input)
            with self.device:
                self.nccl_comm.allReduce(input.data.ptr, input.data.ptr, nccl_size, nccl_dtype,
                                         nccl.NCCL_SUM, cp.cuda.Stream.null.ptr)

        def reduce(self, input, root=0):
            if self.device != get_device(input):
                raise ValueError('Input device is different from communicator device.')

            if self.size == 1:
                return

            nccl_dtype, nccl_size = self._get_nccl_dtype_size(input)
            with self.device:
                self.nccl_comm.reduce(input.data.ptr, input.data.ptr, nccl_size, nccl_dtype,
                                      nccl.NCCL_SUM, root, cp.cuda.Stream.null.ptr)

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
                raise ValueError('dtype not supported, got {dtype}.'.format(dtype=input.dtype))

            return nccl_dtype, nccl_size
