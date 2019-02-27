# Communicator Benchmark for Chainer

Collection of microbenchmarks for
[Chainer](https://www.chainer.org). Originally developed by Takuya
Akiba and maintained by developers.

- comm_bench1: Microbenchmark to test Chainer's communicators'
  `allreduce_grad` performance stats, with ResNet gradients
- comm_bench2: Microbenchmark to test Chainer's communicators'
  `allreduce_grad` performance in timeseries plot, with ResNet gradients

## Usage example & Prerequisites

Requires Python >= 3.5 and MPI which is supported by Chainer.

Install

```sh
$ git clone git://github.com/chainer/comm_bench.git
...
$ cd comm_bench
$ pip install --user .
```

Usage example

```sh
$ mpirun --np 8 -- comm_bench1 --communicator_names pure_nccl,hierarchial
...
$ mpirun --np 16 -hostfile hosts.txt -- comm_bench2 --interval 0.2
...
```

## License

MIT License, (C) 2019 Preferred Networks, Inc.
