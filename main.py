from Kohonen import Kohonen
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-n', '--iterations', help='iteration numbers', default=10000)
parser.add_argument('-c', '--colors', help='number of colors', default=1600)
parser.add_argument('-s', '--save', help='save image ?', default=True)
parser.add_argument('-l', '--learnrate', help='save image ?', default=0.01)
parser.add_argument('-x', '--xdim', help='x dimension of network', default=40)
parser.add_argument('-y', '--ydim', help='y dimension of network', default=40)
args = parser.parse_args()

raw_data = np.random.randint(0, 255, (3, int(args.colors)))
network = Kohonen(data=raw_data, iterations=int(args.iterations), learning_rate=float(args.learnrate), network_x_dimension=int(args.xdim), network_y_dimension=int(args.ydim))
network.normalize()
network.train()
network.show(save=args.save)
