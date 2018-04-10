import argparse
import logging
import math
import os
import time
import collections
import editdistance
import kaldi_io
import mxnet as mx
import numpy as np
from mxnet import autograd, gluon
from mxnet.gluon import contrib
from utils import Accuracy, TokenAcc, convert
from utils import mean, var, dset
from joint_model import Transducer


parser = argparse.ArgumentParser(description='MXNet Autograd RNN/LSTM Acoustic Model on TIMIT.')
parser.add_argument('--lr', type=float, default=1e-3,
                    help='initial learning rate')
parser.add_argument('--clip', type=float, default=0.2,
                    help='gradient clipping')
parser.add_argument('--epochs', type=int, default=200,
                    help='upper epoch limit')
parser.add_argument('--batch_size', type=int, default=1, metavar='N',
                    help='batch size')
parser.add_argument('--dropout', type=float, default=0.5,
                    help='dropout applied to layers (0 = no dropout)')
parser.add_argument('--log-interval', type=int, default=50, metavar='N',
                    help='report interval')
parser.add_argument('--out', type=str, default='exp/rnnt_init',
                    help='path to save the final model')
parser.add_argument('--init', type=str, default='',
                    help='Initial am & pm parameters')
parser.add_argument('--initam', type=str, default='',
                    help='Initial am parameters')
parser.add_argument('--initpm', type=str, default='',
                    help='Initial pm parameters')
parser.add_argument('--schedule', default=False, action='store_true')
args = parser.parse_args()

os.makedirs(args.out, exist_ok=True)
logging.basicConfig(format='%(asctime)s: %(message)s', datefmt="%m-%d %H:%M:%S", filename=os.path.join(args.out, 'train.log'), level=logging.INFO)

context = mx.gpu(0)
# Dataset
dset = dset()
train_feat = dset['train']['feat']
train_label = dset['train']['label']
cv_feat = dset['cv']['feat']
cv_label = dset['cv']['label']

###############################################################################
# Build the model
###############################################################################

model = Transducer(dropout=args.dropout)
# model.collect_params().initialize(mx.init.Xavier(), ctx=context)
if args.init:
    model.collect_params().load(args.init, context)
elif args.initam or args.initpm:
    model.initialize(mx.init.Uniform(0.1), ctx=context)
    if args.initam:
        model.collect_params('transducer0_rnnmodel0').load(args.initam, context, True, True)
    if args.initpm:
        model.collect_params('transducer0_rnnmodel1').load(args.initpm, context, True, True)
    # model.collect_params().save(args.out+'/init')
    # print('initial model save to', args.out+'/init')
else:
    model.initialize(mx.init.Uniform(0.1), ctx=context)
    
trainer = gluon.Trainer(model.collect_params(), 'sgd',
                        {'learning_rate': args.lr,
                         'momentum': 0.9})

###############################################################################
# Training code
###############################################################################

def evaluate(ctx=context):
    losses = []
    for i, (k, v) in enumerate(kaldi_io.read_mat_ark(cv_feat)):
        xs, ys, xlen, ylen = convert((v, cv_label[k]), ctx)
        loss = model(xs, ys, xlen, ylen)
        losses.append(float(loss.sum().asscalar()))
    return sum(losses) / len(losses) / args.batch_size

def train():
    best_model = None
    prev_loss = 1000
    for epoch in range(1, args.epochs):
        losses = []
        totl0 = 0
        start_time = time.time()
        for i, (k, v) in enumerate(kaldi_io.read_mat_ark(train_feat)):
            xs, ys, xlen, ylen = convert((v, train_label[k]), context)
            with autograd.record():
                loss = model(xs, ys, xlen, ylen)
                loss.backward()

            losses.append(float(loss.sum().asscalar()))
            # gradient clip
            # grads = [p.grad(context) for p in model.collect_params().values()]
            # gluon.utils.clip_global_norm(grads, 5)

            trainer.step(args.batch_size, ignore_stale_grad=True)
            totl0 += losses[-1]

            if i % args.log_interval == 0 and i > 0:
                l0 = totl0 / args.batch_size / args.log_interval
                logging.info('[Epoch %d Batch %d] loss %.2f'%(epoch, i, l0))
                totl0 = 0

        losses = sum(losses) / len(losses) / args.batch_size
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