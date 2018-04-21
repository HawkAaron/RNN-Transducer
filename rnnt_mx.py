import mxnet as mx
from mxnet import autograd, gluon

class RNNTLoss(gluon.loss.Loss):
    def __init__(self, blank_label=0, weight=None, **kwargs):
        super(RNNTLoss, self).__init__(weight, batch_axis=0, **kwargs)
        self.blank_label = blank_label
        
    def hybrid_forward(self, F, pred, label,
                pred_lengths=None, label_lengths=None):
        cpu = mx.cpu()
        loss = F.contrib.RNNTLoss(pred.as_in_context(cpu), label.as_in_context(cpu), 
            pred_lengths.as_in_context(cpu), label_lengths.as_in_context(cpu),
                        blank_label=self.blank_label)
        return loss
