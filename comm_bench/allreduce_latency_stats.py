#!/usr/bin/env python3
import argparse
import os
import sys

import matplotlib.pyplot as plt
import pandas as pd


def try_read(path, rank):
    filename = os.path.join(path, "latency.{}.log".format(rank))
    print('Reading', filename, '...', end='')
    df = pd.read_csv(filename, header=[0]) #names=[str(rank)])
    print('found.')
    return df

def read_latency_trace(path):
    rank = 0
    dfs = []
    while True:
        try:
            df = try_read(path, rank)
            dfs.append(df)
            rank += 1
        except FileNotFoundError:
            print('not found.')
            break
    df = pd.concat(dfs, axis=1)
    return df

def heatmap(df, out, **kwargs):
    vmax = kwargs.get('vmax', 5000)
    hm = plt.pcolor(df.values, cmap='hot', vmin=0, vmax=vmax)
    plt.colorbar(hm, label='latency (ms)')
    title = kwargs.get('title')
    if title is None:
        title = 'Latency heatmap on time/rank'
    plt.title(title)
    plt.ylabel('allreduce round')
    plt.xlabel('rank')
    plt.gca().invert_yaxis()
    plt.savefig(out)

def main():
    '''Computes the elapsed time between two events (in milliseconds with
    a resolution of around 0.5 microseconds).

    https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__EVENT.html#group__CUDART__EVENT_1g40159125411db92c835edb46a0989cd6
    '''

    parser = argparse.ArgumentParser('Toolkit to treat ChainerMN AllReduce latency trace')
    parser.add_argument('--path', help="directory to read", default='result')
    parser.add_argument('--out', help="filename to save", default=None)
    parser.add_argument('--title', help="Title of the graph", default=None)
    parser.add_argument('--vmax', help='vmax for heatmap in ms', default=5000)
    # TODO(kuenishi): Add argument to ignore several ranks where files may lack
    args = parser.parse_args()
    df = read_latency_trace(args.path)

    if args.out is None:
        print(df)
        return

    ext = os.path.splitext(args.out)[1]

    if ext == '.pq':
        df.to_parquet(args.out)
        print("Saved to", args.out)
    elif ext == '.csv':
        df.to_csv(args.out, index=False)
        print("Saved to", args.out)
    elif ext == '.svg' or ext == '.png':
        heatmap(df, args.out, title=args.title, vmax=args.vmax)
        print("Heatmap saved to", args.out)
    else:
        print(df)

    sys.exit(0)

if __name__ == '__main__':
    main()
