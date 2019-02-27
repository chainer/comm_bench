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

    from chainercv.links import ResNet101
    from chainercv.links import ResNet152
    from chainercv.links import ResNet50

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


possible_comm_names = ['flat', 'hierarchical', 'two_dimensional', 'naive', 'pure_nccl', 'pure_nccl_fp16']
default_com_names = ['pure_nccl', 'pure_nccl_fp16']
    
def setup_comm(name):
    kwargs = {}
    if name == 'pure_nccl_fp16':
        kwargs['allreduce_grad_dtype'] = np.float16
    comm = chainermn.create_communicator(name, **kwargs)
    return comm

from . import setup_comm, setup_model, update_once

def main():
    parser = argparse.ArgumentParser('Communicator Benchmark 1: allreduce_grad latency stats')
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
    args = parser.parse_args()
    
    model = setup_model(args.model, args.label_num)
    communicator_names = args.communicator_names
    use_gpu = args.use_gpu
    model_name = args.model
    n_trials = args.n_trials

    if args.communicator_names is None:
        communicator_names = default_com_names
    for name in communicator_names:
        assert name in possible_comm_names

    mpi_comm = mpi4py.MPI.COMM_WORLD
    cuda_stream = chainer.cuda.Stream.null
    comm = chainermn.create_communicator('hierarchical')

    if use_gpu:
        chainer.cuda.get_device(comm.intra_rank).use()
        model.to_gpu()
    update_once(model)

    import socket
    hosts =  comm.gather_obj((comm.rank, socket.gethostname()))

    if comm.rank == 0:
        print('Process map:', hosts)
        print("Communicator names:", communicator_names)

        logger.info('Workers: Total={}, Inter={}, Intra={}'.format(
            comm.size, comm.inter_size, comm.intra_size))

        n_elems_total = sum(param.grad.size for param in model.params())
        n_bytes_total = n_elems_total * 4
        logger.info('Model: {} ({} params, {} bytes)'.format(
            model_name, len(list(model.params())), n_bytes_total))
        logger.info('Trials: {}'.format(n_trials))
        logger.info('-----------------------------------------------')
        logger.info('Communicator     Mean    Median  Min     Max     Std')

    for communicator_name in communicator_names:
        comm = setup_comm(communicator_name)

        times = []
        for trial in range(n_trials + 1):
            cuda_stream.synchronize()
            mpi_comm.Barrier()

            time_start = time.time()
            comm.allreduce_grad(model)
            cuda_stream.synchronize()
            mpi_comm.Barrier()

            if trial > 0:
                times.append(time.time() - time_start)

        if comm.rank == 0:
            times = np.asarray(times)
            logger.info('{:<15}{:>8.4f}{:>8.4f}{:>8.4f}{:>8.4f}{:>8.4f}'.format(
                communicator_name, np.mean(times), np.median(times),
                np.min(times), np.max(times), np.std(times)))
            # TODO: plot histogram here?

if __name__ == '__main__':
    try:
        main()
    except:
        import socket
        print(mpi4py.MPI.COMM_WORLD.rank, socket.gethostname())
        import traceback
        traceback.print_exc()
