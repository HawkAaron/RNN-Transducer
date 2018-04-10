import argparse
import logging
import math
import os
import time

import editdistance
import kaldi_io
import mxnet as mx
import numpy as np

from utils import TokenAcc
from utils import mean, var, rephone, pmap, dset
from joint_model import Transducer

parser = argparse.ArgumentParser(description='MXNet Autograd RNN/LSTM Acoustic Model on TIMIT.')
parser.add_argument('model', help='trained model filename')
parser.add_argument('--beam', type=int, default=0)
parser.add_argument('--dataset', default='test')
parser.add_argument('--out', type=str, default='', help='decoded result output dir')
args = parser.parse_args()

logdir = args.out if args.out else os.path.dirname(args.model) + '/decode.log'
logging.basicConfig(format='%(asctime)s: %(message)s', datefmt="%H:%M:%S", filename=logdir, level=logging.INFO)

context = mx.cpu(0)

# Load model
model = Transducer()
model.collect_params().load(args.model, context)
# Dataset
dset = dset()

def remap(y, blank=0):
    prev = blank
    seq = []
    for i in y:
        if i != blank and i != prev: seq.append(i)
        prev = i
    return seq

def distance(y, t, blank=0):
    y = remap(y, blank)
    t = remap(t, blank)
    return y, t, editdistance.eval(y, t)

def decode(dtype='test'):
    logging.info('Decoding Transduction model:')
    feat = dset[dtype]['feat']
    label = dset[dtype]['label']
    err = cnt = 0
    for i, (k, v) in enumerate(kaldi_io.read_mat_ark(feat)):
        xs = mx.nd.array((v[None, ...]+mean)*var)
        if args.beam > 0:
            y = model.beam_search(xs, args.beam)
        else:
            y = model.greedy_decode(xs)
        y = [pmap[i] for i in y]
        t = label[k]
        t = [pmap[i] for i in t]
        y, t, e = distance(y, t)
        err += e
        cnt += len(t)
        y = [rephone[i] for i in y]
        t = [rephone[i] for i in t]
        logging.info('[{}]: {}'.format(k, ' '.join(t)))
        logging.info('[{}]: {}\n'.format(k, ' '.join(y)))
    logging.info('{} set Transducer PER {:.2f}%\n'.format(dtype.capitalize(), 100*err/cnt))
    
decode(args.dataset)