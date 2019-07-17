from . import CommBench, setup_model, update_once, print_experimental_env

import argparse
import socket

from logging import DEBUG
from logging import getLogger
from logging import StreamHandler

logger = getLogger(__name__)
handler = StreamHandler()
handler.setLevel(DEBUG)
logger.setLevel(DEBUG)
logger.addHandler(handler)

try:
    import chainer
    import chainermn
    import mpi4py.MPI

except:
    logger.exception("rank=%d, host=%s", mpi4py.MPI.COMM_WORLD.rank,
                     socket.gethostname())
    import traceback
    traceback.print_exc()
    raise

possible_comm_names = ['flat', 'hierarchical', 'two_dimensional',
                       'naive', 'pure_nccl', 'pure_nccl_fp16']
default_com_names = ['pure_nccl', 'pure_nccl_fp16']


def main():
    parser = argparse.ArgumentParser('Communicator Benchmark 1:'
                                     ' allreduce_grad latency stats')
    parser.add_argument('--model', default='resnet50', help="Type of model",
                        choices=['resnet50', 'resnet101', 'resnet152'])
    parser.add_argument('--communicator_names', type=str,
                        help="Communicator names")
    parser.add_argument('--use_gpu', '-g', help='Use GPU',
                        action='store_true', default=True)
    parser.add_argument('--label_num', help='Number of labels',
                        type=int, default=1000)
    parser.add_argument('--n_trials', '-n', help='Number of trials',
                        type=int, default=100)
    parser.add_argument('--interval', help='interval in sec',
                        type=int, default=0)
    parser.add_argument('--verbose', '-v', action='store_true',
                        help="Verbose mode to see each latency")
    args = parser.parse_args()

    model = setup_model(args.model, args.label_num)
    communicator_names = args.communicator_names
    use_gpu = args.use_gpu
    model_name = args.model
    n_trials = args.n_trials

    if args.communicator_names is None:
        communicator_names = default_com_names
    else:
        communicator_names = communicator_names.split(',')
    for name in communicator_names:
        assert name in possible_comm_names

    comm = chainermn.create_communicator()

    if use_gpu:
        chainer.cuda.get_device(comm.intra_rank).use()
        model.to_gpu()
    update_once(model)
    print_experimental_env(comm, model, model_name, n_trials, logger)

    if comm.rank == 0:
        logger.info("Communicator names: {}".format(communicator_names))

        logger.info('-----------------------------------------------')
        logger.info('Communicator     Mean    Median  Min     Max     Std')

    for communicator_name in communicator_names:
        bench = CommBench(communicator_name, n_trials,
                          args.interval, args.verbose)
        bench.benchmark(model)
        bench.pp_result()


if __name__ == '__main__':
    try:
        main()
    except:
        logger.exception("rank=%d, host=%s", mpi4py.MPI.COMM_WORLD.rank,
                         socket.gethostname())
        import traceback
        traceback.print_exc()
