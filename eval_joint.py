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
from ctc_decoder import decode as beam_search

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

def greedy_decode(dtype='test'):
    logging.info('Greedy decode CTC model:')
    feat = dset[dtype]['feat']
    label = dset[dtype]['label']
    err = cnt = 0
    for i, (k, v) in enumerate(kaldi_io.read_mat_ark(feat)):
        xs = mx.nd.array((v[None, ...]+mean)*var)
        y = model.encoder(xs)
        y = mx.nd.argmax(y[0], axis=1).asnumpy()
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
    logging.info('{} set PER {:.2f}%\n'.format(dtype.capitalize(), 100*err/cnt))

# TODO batch decode
def decode_pm(dtype='test'):
    logging.info('Decoding phoneme model:')
    feat = dset[dtype]['feat']
    label = dset[dtype]['label']
    acc = cnt = 0
    for i, (k, v) in enumerate(kaldi_io.read_mat_ark(feat)):
        ys = mx.nd.array(label[k][None, ...])
        h = mx.nd.one_hot(ys-1, model.vocab_size-1) # pm input size 
        h = mx.nd.concat(mx.nd.zeros((h.shape[0], 1, h.shape[2]), ctx=h.context), h, dim=1) # concat zero vector
        g = model.decoder(h)
        g = mx.nd.argmax(g[0], axis=1).asnumpy()
        ys = ys.asnumpy()[0]
        acc += sum(g == np.concatenate((ys, np.zeros((1)))))
        cnt += g.shape[0]
    logging.info('{} set Prediction PER {:.2f}%\n'.format(dtype.capitalize(), 100 - 100*acc/cnt))

def decode_am(dtype='test'):
    logging.info('Beam serach CTC model:')
    feat = dset[dtype]['feat']
    label = dset[dtype]['label']
    err = cnt = 0
    for i, (k, v) in enumerate(kaldi_io.read_mat_ark(feat)):
        xs = mx.nd.array((v[None, ...]+mean)*var)
        y = model.encoder(xs)[0]
        y = mx.nd.log_softmax(y, axis=1)
        y, logp = beam_search(y.asnumpy())
        y = [pmap[i] for i in y]
        t = [pmap[i] for i in label[k]]
        y, t, e = distance(y, t)
        err += e; cnt += len(t)
        y = [rephone[i] for i in y]
        t = [rephone[i] for i in t]
        logging.info('[{}]: {}'.format(k, ' '.join(t)))
        logging.info('[{}]: {} \tlogliklihood: {:.2f}\n'.format(k, ' '.join(y), logp))
    logging.info('{} set CTC PER {:.2f}%\n'.format(dtype.capitalize(), 100*err/cnt))

def decode_rnnt(dtype='test'):
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

def decode(dtype, only=True):
    if not only:
        decode_pm(dtype)
        decode_am(dtype)
    decode_rnnt(dtype)

decode(args.dataset)