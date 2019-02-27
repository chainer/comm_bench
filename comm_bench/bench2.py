import argparse

try:
    from logging import DEBUG
    from logging import getLogger
    from logging import StreamHandler
    import time

    import chainer
    import chainermn
    import mpi4py.MPI
    import numpy as np

    logger = getLogger(__name__)
    handler = StreamHandler()
    handler.setLevel(DEBUG)
    logger.setLevel(DEBUG)
    logger.addHandler(handler)

except:
    import socket
    print(mpi4py.MPI.COMM_WORLD.rank, socket.gethostname())
    import traceback
    traceback.print_exc()

from . import setup_comm, setup_model, update_once
from tqdm import tqdm
import sys

possible_comm_names = ['flat', 'hierarchical', 'two_dimensional', 'naive', 'pure_nccl', 'pure_nccl_fp16']

def main():
    parser = argparse.ArgumentParser('Communicator Benchmark 1: allreduce_grad latency stats')
    parser.add_argument('--model', default='resnet50', help="Type of model",
                        choices=['resnet50', 'resnet101', 'resnet152'])
    parser.add_argument('--communicator_name', type=str,
                        help="Communicator name", default='pure_nccl')
    parser.add_argument('--use_gpu', '-g', help='Use GPU',
                        action='store_true', default=True)
    parser.add_argument('--label_num', help='Number of labels',
                        type=int, default=1000)
    parser.add_argument('--n_trials', '-n', help='Number of trials',
                        type=int, default=100)
    parser.add_argument('--interval', help='Interval in seconds',
                        type=float, default=0)
    parser.add_argument('--plot', help="Plot output filename",
                        default='plot.png')
    args = parser.parse_args()
    
    model = setup_model(args.model, args.label_num)
    label_num = args.label_num
    communicator_name = args.communicator_name
    use_gpu = args.use_gpu
    model_name = args.model
    n_trials = args.n_trials
    interval = args.interval

    model = setup_model(model_name, label_num)

    mpi_comm = mpi4py.MPI.COMM_WORLD
    cuda_stream = chainer.cuda.Stream.null
    comm = setup_comm(communicator_name)

    if use_gpu:
        chainer.cuda.get_device(comm.intra_rank).use()
        model.to_gpu()
    update_once(model)

    import socket
    hosts = comm.gather_obj({comm.rank: socket.gethostname()})

    if comm.rank == 0:
        print("Communicator name:", communicator_name)
        print("Interval(sec):", interval)

        print(hosts)

        logger.info('Workers: Total={}, Inter={}, Intra={}'.format(
            comm.size, comm.inter_size, comm.intra_size))

        n_elems_total = sum(param.grad.size for param in model.params())
        n_bytes_total = n_elems_total * 4
        logger.info('Model: {} ({} params, {} bytes)'.format(
            model_name, len(list(model.params())), n_bytes_total))
        logger.info('Trials: {}'.format(n_trials))

    comm.allreduce_grad(model)

    times = []
    pbar = tqdm(range(n_trials))
    for trial in pbar:
        cuda_stream.synchronize()
        mpi_comm.Barrier()

        time_start = time.time()
        comm.allreduce_grad(model)
        cuda_stream.synchronize()
        mpi_comm.Barrier()
        time_end = time.time()
        pbar.set_description("{}:\tstart={}, duration={}sec".format(trial, time_start, time_end - time_start))

        times.append((time_start, time_end - time_start))
        time.sleep(interval)

    if comm.rank == 0:
        import matplotlib.pyplot as plt
        x, y = zip(*times)
        plt.plot(x, y, '-')
        plt.xlabel('time (sec)')
        plt.ylabel('latency (sec)')
        plt.title('latencies of allreduce_grad by time plot')
        plt.savefig(args.plot)


if __name__ == '__main__':
    try:
        main()
    except:
        import socket
        print(mpi4py.MPI.COMM_WORLD.rank, socket.gethostname())
        import traceback
        traceback.print_exc()
