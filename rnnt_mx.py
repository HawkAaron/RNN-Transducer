from mxnet import autograd, gluon

class RNNTLoss(gluon.loss.Loss):
    def __init__(self, blank_label=0, weight=None, **kwargs):
        super(RNNTLoss, self).__init__(weight, batch_axis=0, **kwargs)
        self.blank_label = blank_label
        
    def hybrid_forward(self, F, pred, label,
                pred_lengths=None, label_lengths=None):
        loss = F.contrib.RNNTLoss(pred, label, pred_lengths, label_lengths,
                        blank_label=self.blank_label)
        return loss
