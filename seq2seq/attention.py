import mxnet as mx
from mxnet import gluon

class Attention(gluon.Block):
    def __init__(self, **kwargs):
        super(Attention, self).__init__(**kwargs)

    def forward(self, hidden, enc_out):
        # BTH * B1H
        output = enc_out * hidden.expand_dims(1) 
        output = output.sum(axis=2) # BT
        output = mx.nd.softmax(output, axis=1)
        # BTH * BT1
        output = enc_out * output.expand_dims(axis=2)
        output = output.sum(axis=1) # BH
        return output