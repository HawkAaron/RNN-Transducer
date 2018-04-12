import mxnet as mx
from mxnet import gluon
from mxnet.gluon import nn, rnn
import numpy as np
from rnnt_np import RNNTLoss

class RNNModel(gluon.Block):
    """A model with an encoder, recurrent layer, and a decoder."""

    def __init__(self, vocab_size, num_hidden, num_layers, dropout=0, bidirectional=False):
        super(RNNModel, self).__init__()
        with self.name_scope():
            self.rnn = rnn.LSTM(num_hidden, num_layers, 'NTC', dropout=dropout, bidirectional=bidirectional)
            if bidirectional: num_hidden *= 2
            self.decoder = nn.Dense(vocab_size, flatten=False, in_units=num_hidden)

    def forward(self, xs):
        h = self.rnn(xs)
        return self.decoder(h)

class Transducer(gluon.Block):
    ''' When joint training, remove RNNModel decoder layer '''
    def __init__(self, vocab_size, num_hidden, num_layers, dropout=0, blank=0, bidirectional=False):
        super(Transducer, self).__init__()
        self.num_hidden = num_hidden
        self.num_layers = num_layers
        self.vocab_size = vocab_size
        self.loss = RNNTLoss(blank)
        self.blank = blank
        with self.name_scope():
            # acoustic model
            self.encoder = RNNModel(vocab_size, num_hidden, num_layers, dropout, bidirectional)
            # prediction model
            self.decoder = RNNModel(vocab_size, num_hidden, 1, dropout)
            # joint 
            input_size = 3*num_hidden if bidirectional else 2*num_hidden
            self.fc1 = nn.Dense(num_hidden, flatten=False, in_units=input_size)
            self.fc2 = nn.Dense(vocab_size, flatten=False, in_units=num_hidden)
    
    def joint(self, f, g):
        ''' `f`: encoder lstm output (B,T,U,2H) expanded
        `g`: decoder lstm output (B,T,U,H) expanded
        NOTE f and g must have the same size except the last dim '''
        dim = len(f.shape) - 1
        out = mx.nd.concat(f, g, dim=dim)
        out = mx.nd.relu(self.fc1(out))
        return self.fc2(out)

    def forward(self, xs, ys, xlen, ylen):
        # forward acoustic model
        f = self.encoder.rnn(xs)
        # forward prediction model
        ymat = mx.nd.one_hot(ys-1, self.vocab_size-1) # pm input size 
        ymat = mx.nd.concat(mx.nd.zeros((ymat.shape[0], 1, ymat.shape[2]), ctx=ymat.context), ymat, dim=1) # concat zero vector
        g = self.decoder.rnn(ymat)
        # rnnt loss
        f1 = mx.nd.expand_dims(f, axis=2) # BT1H
        g1 = mx.nd.expand_dims(g, axis=1) # B1UH
        f1 = mx.nd.broadcast_axis(f1, 2, g1.shape[2])
        g1 = mx.nd.broadcast_axis(g1, 1, f1.shape[1])
        ytu = mx.nd.softmax(self.joint(f1, g1), axis=3)
        loss = self.loss(ytu, ys, xlen, ylen)
        return loss
    
    def greedy_decode(self, xs):
        '''
        TODO batch support / gpu support
        `weight`: acoustic score weight
        '''
        # forward acoustic model TODO streaming decode
        h = self.encoder.rnn(xs)[0]
        y = mx.nd.zeros((1, 1, self.vocab_size-1)) # first zero vector 
        hid = [mx.nd.zeros((1, 1, self.num_hidden))] * 2 # support for one sequence
        y, hid = self.decoder.rnn(y, hid) # forward first zero
        y_seq = []; logp = 0
        for xi in h:
            ytu = self.joint(y[0][0], xi)
            ytu = mx.nd.log_softmax(ytu)
            yi = mx.nd.argmax(ytu, axis=0) # for Graves2012 transducer
            pred = int(yi.asscalar()); logp += float(ytu[yi].asscalar())
            if pred != self.blank:
                y_seq.append(pred)
                y = mx.nd.one_hot(yi.reshape((1,1))-1, self.vocab_size-1)
                y, hid = self.decoder.rnn(y, hid) # forward first zero
        return y_seq, -logp

    def beam_search(self, xs, W=10, prefix=True):
        '''
        `xs`: acoustic model outputs
        NOTE only support one sequence (batch size = 1)
        '''
        def forward_step(label, hidden):
            ''' `label`: int '''
            label = mx.nd.one_hot(mx.nd.full((1,1), label-1, dtype=np.int32), self.vocab_size-1)
            pred, hidden = self.decoder.rnn(label, hidden)
            return pred[0][0], hidden

        def isprefix(a, b):
            # a is the prefix of b
            if a == b or len(a) >= len(b): return False
            for i in range(len(a)):
                if a[i] != b[i]: return False
            return True

        F = mx.nd
        xs = self.encoder.rnn(xs)[0]
        B = [Sequence(blank=self.blank, hidden=[mx.nd.zeros((1, 1, self.num_hidden))] * 2)]
        for i, x in enumerate(xs):
            if prefix: sorted(B, key=lambda a: len(a.k), reverse=True) # larger sequence first add
            A = B
            B = []
            if prefix:
                # for y in A:
                #     y.logp = log_aplusb(y.logp, prefixsum(y, A, x))
                for j in range(len(A)-1):
                    for i in range(j+1, len(A)):
                        if not isprefix(A[i].k, A[j].k): continue
                        # A[i] -> A[j]
                        pred, _ = forward_step(A[i].k[-1], A[i].h)
                        idx = len(A[i].k)
                        ytu = self.joint(pred, x)
                        logp = F.log_softmax(ytu).asnumpy()
                        curlogp = A[i].logp + float(logp[A[j].k[idx]])
                        for k in range(idx, len(A[j].k)-1):
                            ytu = self.joint(A[j].g[k], x)
                            logp = F.log_softmax(ytu, axis=0)
                            curlogp += float(logp[A[j].k[k+1]].asscalar())
                        A[j].logp = log_aplusb(A[j].logp, curlogp)

            while True:
                y_hat = max(A, key=lambda a: a.logp)
                # y* = most probable in A
                # y_hat = A[0]
                # remove y* from A
                # A = A[1:]
                A.remove(y_hat)
                # calculate P(k|y_hat, t)
                # get last label and hidden state
                pred, hidden = forward_step(y_hat.k[-1], y_hat.h)
                logp = F.log_softmax(self.joint(pred, x)).asnumpy() # log probability for each k
                # for k \in vocab
                for k in range(self.vocab_size):
                    yk = Sequence(y_hat)
                    yk.logp += float(logp[k])
                    if k == self.blank:
                        B.append(yk) # next move
                        continue
                    # store prediction distribution and last hidden state
                    # yk.h.append(hidden); yk.k.append(k)
                    yk.h = hidden; yk.k.append(k); 
                    if prefix: yk.g.append(pred)
                    A.append(yk)
                # sort A
                # sorted(A, key=lambda a: a.logp, reverse=True) # just need to calculate maximum seq
                
                # sort B
                # sorted(B, key=lambda a: a.logp, reverse=True)
                y_hat = max(A, key=lambda a: a.logp)
                yb = max(B, key=lambda a: a.logp)
                if len(B) >= W and yb.logp >= y_hat.logp: break

            # beam width
            sorted(B, key=lambda a: a.logp, reverse=True)
            B = B[:W]

        # return highest probability sequence
        print(B[0])
        return B[0].k, -B[0].logp

import math
def log_aplusb(a, b):
    return max(a, b) + math.log1p(math.exp(-math.fabs(a-b)))

from utils import rephone
class Sequence():
    def __init__(self, seq=None, hidden=None, blank=0):
        if seq is None:
            self.g = [] # predictions of phoneme language model
            self.k = [blank] # prediction phoneme label
            # self.h = [None] # input hidden vector to phoneme model
            self.h = hidden
            self.logp = 0 # probability of this sequence, in log scale
        else:
            self.g = seq.g[:] # save for prefixsum
            self.k = seq.k[:]
            self.h = seq.h
            self.logp = seq.logp

    def __str__(self):
        return 'Prediction: {}\nlog-likelihood {:.2f}\n'.format(' '.join([rephone[i] for i in self.k]), -self.logp)