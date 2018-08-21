import numpy as np
from sigpy import config, util
if config.cupy_enabled:
    import cupy as cp
    
if config.mpi4py_enabled:
    from mpi4py import MPI

    if config.nccl_enabled:
        from cupy.cuda import nccl
    

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
            mpi_buffer = util.move(input)
            self.mpi_comm.Allreduce(MPI.IN_PLACE, mpi_buffer)
            util.move_to(input, mpi_buffer)


class MultiGpuCommunicator(object):
    """Communicator for distributed computing between multiple GPU.

    If nccl is installed with cupy, then nccl will be used. Otherwise,
    reduces to Communicator.

    """

    def __init__(self):
        super().__init__()
        self.device = util.Device(self.rank % cp.cuda.runtime.getDeviceCount())

        if config.nccl_enabled:
            if self.rank == 0:
                nccl_comm_id = nccl.get_unique_id()
            else:
                nccl_comm_id = None

            nccl_comm_id = self.mpi_comm.bcast(nccl_comm_id)

            with self.device:
                self.nccl_comm = nccl.NcclCommunicator(
                    self.size, nccl_comm_id, self.rank)

    def allreduce(self, input):
        if self.device != util.get_device(input):
            raise ValueError('Input device is different from communicator device.')

        if self.size == 1:
            return

        if config.nccl_enabled:
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

            with self.device:
                self.nccl_comm.allReduce(
                    input.data.ptr, input.data.ptr, nccl_size, nccl_dtype,
                    nccl.NCCL_SUM, cp.cuda.Stream.null.ptr)
        else:
            super().allreduce(input)
