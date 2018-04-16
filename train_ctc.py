import argparse
import logging
import math
import os
import time
import mxnet as mx
import numpy as np
from mxnet import autograd, gluon
from DataLoader import SequentialLoader, TokenAcc
from model import RNNModel

# CTC loss NOTE why recode this? cause Gluon force `blank_label` to last!!!
class CTCLoss(gluon.loss.Loss):
    def __init__(self, layout='NTC', label_layout='NT', weight=None, **kwargs):
        assert layout in ['NTC', 'TNC'],\
            "Only 'NTC' and 'TNC' layouts for pred are supported. Got: %s"%layout
        assert label_layout in ['NT', 'TN'],\
            "Only 'NT' and 'TN' layouts for label are supported. Got: %s"%label_layout
        self._layout = layout
        self._label_layout = label_layout
        batch_axis = label_layout.find('N')
        super(CTCLoss, self).__init__(weight, batch_axis, **kwargs)

    def hybrid_forward(self, F, pred, label,
                pred_lengths=None, label_lengths=None):
        if self._layout == 'NTC':
            pred = F.swapaxes(pred, 0, 1)
        if self._batch_axis == 1:
            label = F.swapaxes(label, 0, 1)
        loss = F.contrib.CTCLoss(pred, label, pred_lengths, label_lengths,
                        use_data_lengths=pred_lengths is not None,
                        use_label_lengths=label_lengths is not None,
                        blank_label='first')
        return loss

parser = argparse.ArgumentParser(description='MXNet Autograd RNN/LSTM Acoustic Model on TIMIT.')
parser.add_argument('--lr', type=float, default=1e-3,
                    help='initial learning rate')
parser.add_argument('--epochs', type=int, default=200,
                    help='upper epoch limit')
parser.add_argument('--batch_size', type=int, default=1, metavar='N',
                    help='batch size')
parser.add_argument('--dropout', type=float, default=0.5,
                    help='dropout applied to layers (0 = no dropout)')
parser.add_argument('--bi', default=False, action='store_true', 
                    help='whether use bidirectional lstm')
parser.add_argument('--noise', type=float, default=0, 
                    help='add gaussian noise to inputs')
parser.add_argument('--log-interval', type=int, default=50, metavar='N',
                    help='report interval')
parser.add_argument('--out', type=str, default='exp/ctc_lr1e-3',
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

context = mx.gpu(0)
# Dataset
trainset = SequentialLoader('train', args.batch_size, context)
devset = SequentialLoader('dev', args.batch_size, context)

###############################################################################
# Build the model
###############################################################################

model = RNNModel(62, 250, 3, args.dropout, bidirectional=args.bi)
if args.init: model.collect_params().load(args.init, context)
else: model.initialize(mx.init.Uniform(0.1), ctx=context)

trainer = gluon.Trainer(model.collect_params(), 'sgd',
                        {'learning_rate': args.lr,
                         'momentum': 0.9})
criterion = CTCLoss()
###############################################################################
# Training code
###############################################################################

def evaluate(ctx=context):
    losses = []
    for xs, ys, xlen, ylen in devset:
        out = model(xs)
        loss = criterion(out, ys, xlen, ylen)
        losses.append(float(loss.sum().asscalar()))
    return sum(losses) / len(devset), sum(losses) / devset.label_cnt

def train():
    best_model = None
    prev_loss = 1000
    for epoch in range(1, args.epochs):
        losses = []
        totl0 = 0
        start_time = time.time()
        for i, (xs, ys, xlen, ylen) in enumerate(trainset):
            if args.noise > 0: 
                xs += mx.nd.normal(0, args.noise, xs.shape[-1], ctx=xs.context)

            with autograd.record():
                out = model(xs)
                loss = criterion(out, ys, xlen, ylen)
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
        val_l, logl = evaluate()

        logging.info('[Epoch %d] time cost %.2fs, train loss %.2f; cv loss %.2f, log_loss %.2f; lr %.2e'%(
            epoch, time.time()-start_time, losses, val_l, logl, trainer.learning_rate))

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