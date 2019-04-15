import mxnet as mx
from mxnet import autograd, gluon

class RNNTLoss(gluon.loss.Loss):
    def __init__(self, batch_first=True, blank_label=0, weight=None, **kwargs):
        batch_axis = 0 if batch_first else 2
        super(RNNTLoss, self).__init__(weight, batch_axis, **kwargs)
        self.batch_first = batch_first
        self.blank_label = blank_label

    def hybrid_forward(self, F, pred, label, pred_lengths, label_lengths):
        if not self.batch_first:
            pred = F.transpose(pred, (2, 0, 1, 3))

        loss = F.contrib.RNNTLoss(pred, label.astype('int32', False), 
                                    pred_lengths.astype('int32', False), 
                                    label_lengths.astype('int32', False), 
                                    blank_label=self.blank_label)
        return loss

