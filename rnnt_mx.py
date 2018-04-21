import mxnet as mx
from mxnet import autograd, gluon

class RNNTLoss(gluon.loss.Loss):
    def __init__(self, layout='NTC', label_layout='NT', blank_label=0, weight=None, **kwargs):
        assert layout in ['NTC', 'TNC'],\
            "Only 'NTC' and 'TNC' layouts for pred are supported. Got: %s"%layout
        assert label_layout in ['NT', 'TN'],\
            "Only 'NT' and 'TN' layouts for label are supported. Got: %s"%label_layout
        self._layout = layout
        self._label_layout = label_layout
        batch_axis = label_layout.find('N')
        super(RNNTLoss, self).__init__(weight, batch_axis, **kwargs)
        self.blank_label = blank_label
        
    def hybrid_forward(self, F, pred, label, pred_lengths, label_lengths):
        if self._layout == 'NTC':
            pred = F.moveaxis(pred, 0, 2)
        if self._batch_axis == 1:
            label = F.swapaxes(label, 0, 1)
        cpu = mx.cpu()
        loss = F.contrib.RNNTLoss(pred.as_in_context(cpu), label.as_in_context(cpu), 
            pred_lengths.as_in_context(cpu), label_lengths.as_in_context(cpu),
                        blank_label=self.blank_label)
        return loss
