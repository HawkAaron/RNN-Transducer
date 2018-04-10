import os
import numpy as np
import editdistance
import kaldi_io
import collections
import mxnet as mx

# Compute CMVN stats
train_dir = 'data-fbank/train/'
cv_dir = 'data-fbank/dev/'
test_dir = 'data-fbank/test/'
# Feature
train_feat = 'ark:copy-feats scp:{}/feats.scp ark:- | apply-cmvn --utt2spk=ark:{}/utt2spk scp:{}/cmvn.scp ark:- ark:- |'.format(train_dir, train_dir, train_dir)
cv_feat = 'ark:copy-feats scp:{}/feats.scp ark:- | apply-cmvn --utt2spk=ark:{}/utt2spk scp:{}/cmvn.scp ark:- ark:- |'.format(cv_dir, cv_dir, cv_dir)
test_feat = 'ark:copy-feats scp:{}/feats.scp ark:- | apply-cmvn --utt2spk=ark:{}/utt2spk scp:{}/cmvn.scp ark:- ark:- |'.format(test_dir, test_dir, test_dir)
# copy-feats scp:data-fbank/train_tr90//feats.scp ark:- | apply-cmvn --utt2spk=ark:data-fbank/train_tr90//utt2spk scp:data-fbank/train_tr90//cmvn.scp ark:- ark:-

if not os.path.exists('exp/cmvn_stats'):
    mean = var = cnt = 0
    for k, v in kaldi_io.read_mat_ark(train_feat):
        mean += np.sum(v, axis=0)
        var += np.sum(v*v, axis=0)
        cnt += v.shape[0]
    mean /= cnt
    var = var / cnt - mean ** 2
    # global cmvn
    mean = -mean
    var = 1 / np.sqrt(var)
    with open('exp/cmvn_stats', 'wb') as f:
        np.save(f, [mean, var])
    print(mean, var)

# Load CMVN stats
with open('exp/cmvn_stats', 'rb') as f:
    cmvn = np.load(f)
mean = cmvn[0]
var = cmvn[1]

# Load dict
with open('lang/phones.txt', 'r') as f:
    phone = {'<eps>': 0}
    rephone = {0: '<eps>'}
    for line in f:
        line = line.split()
        phone[line[0]] = int(line[1])
        rephone[int(line[1])] = line[0]
print(phone)

# Phone map
with open('lang/phones.60-48-39.map', 'r') as f:
    pmap = {0:0}
    for line in f:
        line = line.split()
        if len(line) < 3: continue
        pmap[phone[line[1]]] = phone[line[2]]
print(pmap)

# Load label sequence
def load_label(text, att=False):
    ''' `att`: attention `sos` and `eos` '''
    with open(text, 'r') as f:
        label = {}
        for line in f:
            line = line.split()
            label[line[0]] = np.array([phone[i] for i in line[1:]], dtype='i')
            if att:
                label[line[0]] = np.array([phone['<s>']] + [phone[i] for i in line[1:]] + [phone['</s>']], dtype='i') - 1
    return label

train_label = lambda att=False: load_label(train_dir + '/text', att)
cv_label = lambda att=False: load_label(cv_dir + '/text', att)
test_label = lambda att=False: load_label(test_dir + '/text', att)

# Data set structure
dset = lambda att=False: {'train': {'feat': train_feat, 'label': train_label(att)},
        'cv': {'feat': cv_feat, 'label': cv_label(att)},
        'test': {'feat': test_feat, 'label': test_label(att)}}


def zero_pad_concat(inputs):
    max_t = max(inp.shape[0] for inp in inputs)
    shape = (len(inputs), max_t) + inputs[0].shape[1:]
    input_mat = np.zeros(shape, dtype=np.float32)
    for e, inp in enumerate(inputs):
        input_mat[e, :inp.shape[0]] = inp
    return input_mat

def convert(batch, context):
    '''
    `batch`: tuple of acoustic feature and phoneme sequence target
    '''
    inputs, labels = tuple(zip(batch))
    inputs = [(i+mean)*var for i in inputs] # feature transform
    xlen = mx.nd.array([i.shape[0] for i in inputs], ctx=context)
    ylen = mx.nd.array([len(i) for i in labels], ctx=context)
    xs = mx.nd.array(zero_pad_concat(inputs), ctx=context)
    ys = mx.nd.array(zero_pad_concat(labels), ctx=context)
    return xs, ys, xlen, ylen

class Accuracy():
    def __init__(self):
        self.acc = 0
        self.cnt = 0
        self.tmp_acc = 0
        self.tmp_cnt = 0

    def update(self, pred, label):
        # TODO batch support
        e = c = 0
        for p, l in zip(pred, label):
            if len(p.shape) > 1: 
                p = np.argmax(p, axis=1)
            e += sum(p == l)
            c += l.shape[0]
        self.acc += e; self.tmp_acc += e
        self.cnt += c; self.tmp_cnt += c
        return 100 - 100 * e / c

    def get(self, err=True):
        if err: res = 100 - 100 * self.tmp_acc / self.tmp_cnt
        else: res = 100 * self.tmp_acc / self.tmp_cnt
        self.tmp_acc = self.tmp_cnt = 0
        return res

    def getAll(self, err=True):
        if err: return 100 - 100 * self.acc / self.cnt
        else: return 100 * self.acc / self.cnt

class TokenAcc():
    def __init__(self, blank=0):
        self.err = 0
        self.cnt = 0
        self.tmp_err = 0
        self.tmp_cnt = 0
        self.blank = 0
    
    def update(self, pred, label):
        '''
        TODO for batch, need xlen
        label is one dimensinal
        '''
        pred = pred.reshape(-1, pred.shape[-1])
        e = self._distance(pred, label)
        c = label.shape[0]
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

class SchedulePolicy():
    def __init__(self, trainer):
        self.trainer = trainer
        self.lr = trainer.learning_rate
        self.prevs = collections.deque()

    def continuous_big(self, data, fact, n):
        prev = data[0]
        for i in range(1, n):
            if prev < data[i] + fact: return False
        return True

    def update(self, val):
        self.prevs.appendleft(val)
        if len(self.prevs) == 6:
            if self.continuous_big(self.prevs, 1, 4):
                self.lr /= 2
            elif self.continuous_big(self.prevs, 0, 6):
                self.lr /= 2 
            self.trainer.set_learning_rate(self.lr)
            self.prevs.pop(); self.prevs.pop()