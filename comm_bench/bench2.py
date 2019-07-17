import argparse

try:
    from logging import DEBUG
    from logging import getLogger
    from logging import StreamHandler
    import time

    import chainer
    import mpi4py.MPI

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
    raise

from . import setup_comm, setup_model, update_once, print_experimental_env
from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser('Communicator Benchmark 2: '
                                     'allreduce_grad latency stats in'
                                     ' timeseries')
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
    parser.add_argument('--log', help="Log output filename",
                        default='bench2.log')
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

    print_experimental_env(comm, model, model_name, n_trials, logger)

    if comm.rank == 0:
        print("Communicator: ", communicator_name)
        print("Interval(sec):", interval)

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
        pbar.set_description("{}:\tstart={}, duration={}sec"
                             .format(trial, time_start, time_end - time_start))

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

        with open(args.log, 'w') as fp:
            fp.write('trial\tstart\tduration\n')
            for i in range(len(times)):
                start, duration = times[i]
                fp.write('{}\t{}\t{}\n'.format(i, start, duration))
        logger.info("Saved the graph to %s, TSV to %s",
                    args.plot, args.log)


if __name__ == '__main__':
    try:
        main()
    except:
        import socket
        print(mpi4py.MPI.COMM_WORLD.rank, socket.gethostname())
        import traceback
        traceback.print_exc()
