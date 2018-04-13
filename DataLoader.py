import sys
import numpy as np
import kaldi_io
import mxnet as mx

with open('data/lang/phones.txt', 'r') as f:
    phone = {}; rephone = {}
    for line in f:
        line = line.split()
        phone[line[0]] = int(line[1])
        rephone[int(line[1])] = line[0]
print(phone)

def zero_pad_concat(inputs):
    max_t = max(inp.shape[0] for inp in inputs)
    shape = (len(inputs), max_t) + inputs[0].shape[1:]
    input_mat = np.zeros(shape, dtype=np.float32)
    for e, inp in enumerate(inputs):
        input_mat[e, :inp.shape[0]] = inp
    return input_mat

def convert(inputs, labels, ctx):
    # length no need move to gpu
    xlen = mx.nd.array([i.shape[0] for i in inputs])
    ylen = mx.nd.array([i.shape[0] for i in labels])
    xs = mx.nd.array(zero_pad_concat(inputs), ctx=ctx)
    ys = mx.nd.array(zero_pad_concat(labels), ctx=ctx)
    return xs, ys, xlen, ylen

class SequentialLoader:
    def __init__(self, dtype, batch_size=1, ctx=mx.cpu(0)):
        self.ctx = ctx
        self.labels = {}
        self.label_cnt = 0
        self.feats_rspecifier = 'ark:copy-feats scp:data/{}/feats.scp ark:- | apply-cmvn --utt2spk=ark:data/{}/utt2spk scp:data/{}/cmvn.scp ark:- ark:- |\
 add-deltas --delta-order=1 ark:- ark:- | nnet-forward data/final.feature_transform ark:- ark:- |'.format(dtype, dtype, dtype)
        self.batch_size = batch_size
        # load label
        with open('data/'+dtype+'/text', 'r') as f:
            for line in f:
                line = line.split()
                self.labels[line[0]] = np.array([phone[i] for i in line[1:]])
                self.label_cnt += len(self.labels[line[0]])
        # load feature

    def __len__(self):
        return len(self.labels)

    def _dump(self):
        for k, v in kaldi_io.read_mat_ark(self.feats_rspecifier):
            label = self.labels[k]
            with open('data-npy/'+k+'.y', 'wb') as f:
                np.save(f, label)
            with open('data-npy/'+k+'.x', 'wb') as f:
                np.save(f, v)
            print(k)

    def __iter__(self):
        feats = []; label = []
        for k, v in kaldi_io.read_mat_ark(self.feats_rspecifier):
            if len(feats) >= self.batch_size:
                yield convert(feats, label, self.ctx)
                feats = []; label = []
            feats.append(v); label.append(self.labels[k])
        yield convert(feats, label, self.ctx)

import editdistance
class TokenAcc():
    def __init__(self, blank=0):
        self.err = 0
        self.cnt = 0
        self.tmp_err = 0
        self.tmp_cnt = 0
        self.blank = 0
    
    def update(self, pred, label, xlen, ylen):
        pred = np.vstack([pred[i, :j] for i, j in enumerate(xlen)])
        label = np.hstack([label[i, :j] for i, j in enumerate(ylen)])
        e = self._distance(pred, label)
        c = sum(ylen)
        self.tmp_err += e; self.err += e
        self.tmp_cnt += c; self.cnt += c
        return 100 * e / c

    def get(self, err=True):
        # get interval
        if err: res = 100 * self.tmp_err / self.tmp_cnt
        else: res = 100 - 100 * self.tmp_err / self.tmp_cnt
        self.tmp_err = self.tmp_cnt = 0
        return res

    def getAll(self, err=True):
        if err: return 100 * self.err / self.cnt
        else: return 100 - 100 * self.err / self.cnt

    def _distance(self, y, t):
        if len(y.shape) > 1: 
            y = np.argmax(y, axis=1)
        prev = self.blank
        hyp = []
        for i in y:
            if i != self.blank and i != prev: hyp.append(i)
            prev = i
        return editdistance.eval(hyp, t)

if __name__ == '__main__':
    SequentialLoader('train')._dump()
    SequentialLoader('dev')._dump()