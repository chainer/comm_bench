import argparse
import numpy as np
import chainer

import mpi4py.MPI
import timeit
import cupy as cp

import chainermn
from cupy.cuda import nccl
import ctypes



def array_to_buffer_object(array):
    array = xp.ascontiguousarray(array)
    xp = chainer.backend.get_array_module(array)

    if xp is np:
        return array
    else:
        return ctypes.cast(
            array.data.ptr,
            ctypes.POINTER(ctypes.c_ubyte * array.nbytes)
        ).contents


def mpi_allreduce(comm, a):
    comm.mpi_comm.Allreduce(mpi4py.MPI.IN_PLACE, array_to_buffer_object(a))


def nccl_allreduce(comm, a):
    stream = chainer.cuda.Stream.null
    if c.dtype == np.float32:
        t = nccl.NCCL_FLOAT
    elif c.dtype == np.float16:
        t = nccl.NCCL_FLOAT16
    comm.nccl_comm.allreduce(a.data.ptr, a.data.ptr, a.size, t, nccl.NCCL_SUM, stream.ptr)
    stream.synchronize()


def main():
    parser = args.ArgumentParser('Microbench of AllReduce')
    parser.add_argument('--channels', help="Number of channels", default=256)
    parser.add_argument('--trials', help="Number of trials", default=100)
    args = parser.parse_args()
    
    n_channels=args.channels
    n_trials=args.trials

    comm = chainermn.create_communicator('hierarchical')
    chainer.cuda.get_device(comm.intra_rank).use()
    comm._init_comms()

    def bench(name, lmd):
        result = timeit.timeit(lmd, globals=globals(), number=n_trials)
        if comm.rank == 0:
            print('{:<20}{:.3e}'.format(name, result / n_trials))

    if comm.rank == 0:
        print('Workers: Total={}, Inter={}, Intra={}'.format(
            comm.size, comm.inter_size, comm.intra_size))
        print('Elems: {}'.format(n_channels))
        print('Trials: {}'.format(n_trials))
        print('-' * 30)

    a = np.arange(n_channels, dtype=np.float32)
    bench('MPI CPU Allreduce', lambda: mpi_allreduce(comm, a))

    a = cp.arange(n_channels, dtype=np.float32)
    bench('MPI GPU Allreduce', lambda: mpi_allreduce(comm, a))

    # FP32
    comm = chainermn.create_communicator('pure_nccl')
    comm._init_comms()
    a = cp.arange(n_channels, dtype=np.float32)
    bench('NCCL Allreduce @FP32', lambda: nccl_allreduce(comm, a))

    a = cp.arange(n_channels, dtype=np.float16)
    bench('NCCL Allreduce @FP16', lambda: nccl_allreduce(comm, a))
    

if __name__ == '__main__':
    main()
