from enum import Enum
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-e', type=int, help='eccentricity', default=30)
parser.add_argument('-m', type=int, help='mode (1 or 2)', default=1)
parser.add_argument('-gap', type=int, help='use GAP (0 or 1)', default=1)
parser.add_argument('-noise', type=int, help='noise (number of pixels)', default=0)
parser.add_argument('-n', type=int, help='number of examples in training set', default=1000)
parser.add_argument('-pretrained', type=int, help='use pretrained weights', default=1)
parser.add_argument('-s', type=int, help='specify canvas size (default=400)', default=400)
args = parser.parse_args()