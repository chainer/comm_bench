import socket
import time

import chainer
import chainermn
import numpy as np
from chainercv.links import ResNet101
from chainercv.links import ResNet152
from chainercv.links import ResNet50


def setup_comm(name):
    kwargs = {}
    if name == 'pure_nccl_fp16':
        name = 'pure_nccl'
        kwargs['allreduce_grad_dtype'] = np.float16
    comm = chainermn.create_communicator(name, **kwargs)
    return comm


def setup_model(model_name, label_num):
    model_cfgs = {
        'resnet50': {'class': ResNet50, 'score_layer_name': 'fc6',
                     'kwargs': {'arch': 'fb'}},
        'resnet101': {'class': ResNet101, 'score_layer_name': 'fc6',
                      'kwargs': {'arch': 'fb'}},
        'resnet152': {'class': ResNet152, 'score_layer_name': 'fc6',
                      'kwargs': {'arch': 'fb'}}
    }
    assert model_name in model_cfgs.keys()
    model_cfg = model_cfgs[model_name]
    extractor = model_cfg['class'](
        n_class=label_num, **model_cfg['kwargs'])
    extractor.pick = model_cfg['score_layer_name']
    model = chainer.links.Classifier(extractor)
    model.cleargrads()

    return model


def update_once(model):
    opt = chainer.optimizers.MomentumSGD()
    opt.setup(model)
    import cupy as cp
    imgs = cp.ndarray((1, 3, 224, 224), dtype=np.float32)
    labels = cp.ndarray((1,), dtype=np.int32)
    labels += 1
    model(imgs, labels)
    opt.update()


def print_experimental_env(comm, model, model_name, n_trials, logger):
    hosts = comm.gather_obj((comm.rank, socket.gethostname()))
    if comm.rank == 0:
        logger.info('Process map: {}'.format(hosts))
        logger.info('Workers: Total={}, Inter={}, Intra={}'.format(
            comm.size, comm.inter_size, comm.intra_size))

        n_elems_total = sum(param.grad.size for param in model.params())
        n_bytes_total = n_elems_total * 4
        logger.info('Model: {} ({} params, {} bytes)'.format(
            model_name, len(list(model.params())), n_bytes_total))
        logger.info('Trials: {}'.format(n_trials))
        logger.info('-----------------------------------------------')


class CommBench(object):
    def __init__(self, comm_name, n_trials=100, interval=0,
                 verbose=False, comm=None):
        self.comm = setup_comm(comm_name) if comm is None else comm
        self.comm_name = comm_name
        assert n_trials > 0
        self.n_trials = n_trials

        assert interval >= 0
        self.interval = interval
        self.verbose = verbose

    def benchmark(self, model):
        times = []
        mpi_comm = self.comm.mpi_comm
        cuda_stream = chainer.cuda.Stream.null
        n_trials = self.n_trials
        # First allreduce_grad is for bcast_data
        self.comm.allreduce_grad(model)

        for trial in range(n_trials + 1):
            cuda_stream.synchronize()
            mpi_comm.Barrier()

            time_start = time.time()
            self.comm.allreduce_grad(model)
            cuda_stream.synchronize()
            mpi_comm.Barrier()
            time_end = time.time()

            if trial > 0:
                times.append((time_end, time_end - time_start))

            if self.verbose and self.comm.rank == 0:
                print("Run #{}:\t{}".format(trial, time_end - time_start))
            if self.interval:
                time.sleep(self.interval)

        self.times = times

    def pp_result(self):
        _, times = zip(*(self.times))
        if self.comm.rank == 0:
            times = np.asarray(times)
            print('{:<15}{:>8.4f}{:>8.4f}{:>8.4f}{:>8.4f}{:>8.4f}'
                  .format(
                      self.comm_name, np.mean(times), np.median(times),
                      np.min(times), np.max(times), np.std(times)))
            # TODO: plot histogram here?

    def plot_result(self, filename):
        import matplotlib.pyplot as plt
        x, y = zip(*(self.times))
        plt.plot(x, y, '-')
        plt.xlabel('time (sec)')
        plt.ylabel('latency (sec)')
        plt.title('latencies of allreduce_grad by time plot')
        plt.savefig(filename)

    def save_result(self, filename):
        with open(filename, 'w') as fp:
            fp.write('trial\tstart\tduration\n')
            for i in range(len(self.times)):
                start, duration = self.times[i]
                fp.write('{}\t{}\t{}\n'.format(i, start, duration))
