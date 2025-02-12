#!/usr/bin/env python3
#
# Script to compare to result files

import argparse
import csv
import statistics

import math

def geomean(xs):
        return math.exp(math.fsum(math.log(x) for x in xs) / len(xs))

def main():
    parser = argparse.ArgumentParser(description='Benchmarks comparisons')

    parser.add_argument('input', nargs=2,
                        help='Two benchmark result files to compute speedup between (OLD NEW)')

    args = parser.parse_args()

    data = dict()
    for res in args.input:
        with open(res, 'r') as f:
            c = csv.reader(f, delimiter=',')
            data[res] = { r[0].split('-')[0]: float(r[1]) for r in c }

    a = data[args.input[0]]
    b = data[args.input[1]]

    speedups = []
    print("|Benchmark|Speedup")
    print("|--|--")
    for k, v in a.items():
        if not k in b:
            continue

        speedup = a[k] / b[k]
        speedups.append(speedup)

        print("|{}|{:.2f}".format(k, speedup))
    print("|--|--")
    print("|Geomean|{}".format(geomean(speedups)))

if __name__ == "__main__":
    main()
