import argparse
import logging
import math
import os
import time
import collections
import editdistance
import random
import kaldi_io
import mxnet as mx
import numpy as np
from mxnet import autograd, gluon
from mxnet.gluon import contrib
from DataLoader import SequentialLoader, TokenAcc
from seq2seq.seq2seq import Seq2seq

parser = argparse.ArgumentParser(description='MXNet Autograd RNN/LSTM Acoustic Model on TIMIT.')
parser.add_argument('--lr', type=float, default=1e-3,
                    help='initial learning rate')
parser.add_argument('--epochs', type=int, default=100,
                    help='upper epoch limit')
parser.add_argument('--batch_size', type=int, default=1, metavar='N',
                    help='batch size')
parser.add_argument('--dropout', type=float, default=0.5,
                    help='dropout applied to layers (0 = no dropout)')
parser.add_argument('--bi', default=False, action='store_true', 
                    help='whether use bidirectional lstm')
parser.add_argument('--sample_rate', type=float, default=0.4, 
                    help='decoder sampleing rate, only for training')
parser.add_argument('--log-interval', type=int, default=50, metavar='N',
                    help='report interval')
parser.add_argument('--out', type=str, default='exp/att_lr1e-3',
                    help='path to save the final model')
parser.add_argument('--stdout', default=False, action='store_true')
parser.add_argument('--init', type=str, default='',
                    help='Initial am & pm parameters')
parser.add_argument('--gradclip', type=float, default=0)
parser.add_argument('--schedule', default=False, action='store_true')
args = parser.parse_args()

os.makedirs(args.out, exist_ok=True)
with open(os.path.join(args.out, 'args'), 'w') as f:
    f.write(str(args))
if args.stdout: logging.basicConfig(format='%(asctime)s: %(message)s', datefmt="%m-%d %H:%M:%S", level=logging.INFO)
else: logging.basicConfig(format='%(asctime)s: %(message)s', datefmt="%m-%d %H:%M:%S", filename=os.path.join(args.out, 'train.log'), level=logging.INFO)
mx.random.seed(1024)
random.seed(1024)

context = mx.gpu(0)
# Dataset
trainset = SequentialLoader('train', args.batch_size, context, True)
devset = SequentialLoader('dev', args.batch_size, context, True)

###############################################################################
# Build the model
###############################################################################

model = Seq2seq(63, 250, 3, args.dropout, args.sample_rate)
# model.collect_params().initialize(mx.init.Xavier(), ctx=context)
if args.init: model.collect_params().load(args.init, context)
else: model.initialize(mx.init.Uniform(0.1), ctx=context)

trainer = gluon.Trainer(model.collect_params(), 'sgd',
                        {'learning_rate': args.lr,
                         'momentum': 0.9})

###############################################################################
# Training code
###############################################################################

def evaluate(ctx=context):
    losses = []
    for xs, ys, xlen, ylen in devset:
        loss = model(xs, ys)[1]
        losses.append(float(loss.sum().asscalar()))
    return sum(losses) / len(devset)

def train():
    best_model = None
    prev_loss = 1000
    for epoch in range(1, args.epochs):
        losses = []
        totl0 = 0
        start_time = time.time()
        for i, (xs, ys, xlen, ylen) in enumerate(trainset):
            with autograd.record():
                loss = model(xs, ys)[1]
                loss.backward()

            losses.append(float(loss.sum().asscalar()))
            # gradient clip
            if args.gradclip > 0:
                grads = [p.grad(context) for p in model.collect_params().values()]
                gluon.utils.clip_global_norm(grads, args.gradclip)

            trainer.step(args.batch_size, ignore_stale_grad=True)
            totl0 += losses[-1]

            if i % args.log_interval == 0 and i > 0:
                l0 = totl0 / args.batch_size / args.log_interval
                logging.info('[Epoch %d Batch %d] loss %.2f'%(epoch, i, l0))
                totl0 = 0

        losses = sum(losses) / len(trainset)
        val_l = evaluate()

        logging.info('[Epoch %d] time cost %.2fs, train loss %.2f; cv loss %.2f; lr %.2e'%(
            epoch, time.time()-start_time, losses, val_l, trainer.learning_rate))

        if val_l < prev_loss:
            prev_loss = val_l
            best_model = '{}/params_epoch{:03d}_tr{:.2f}_cv{:.2f}'.format(args.out, epoch, losses, val_l) 
            model.collect_params().save(best_model)
        else:
            model.collect_params().save('{}/params_epoch{:03d}_tr{:.2f}_cv{:.2f}_rejected'.format(args.out, epoch, losses, val_l))
            model.collect_params().load(best_model, context)
            if args.schedule:
                trainer.set_learning_rate(trainer.learning_rate / 2)

if __name__ == '__main__':
    train()