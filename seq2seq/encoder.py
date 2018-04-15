import mxnet as mx
from mxnet import gluon
from mxnet.gluon import nn, rnn

class Encoder(gluon.Block):
    def __init__(self, hidden_size, num_layers, dropout, **kwargs):
        super(Encoder, self).__init__(**kwargs)
        with self.name_scope():
            self.rnn = rnn.GRU(hidden_size, num_layers, layout='NTC', dropout=dropout)
            self.num_layers = num_layers
            self.hidden_size = hidden_size

    def forward(self, inputs, hidden=None):
        '''
        `inputs`: (batch, length, input_size)
        `hidden`: Initial hidden state (num_layer, batch_size, hidden_size)
        '''
        if hidden is None:
            hidden = mx.nd.zeros((self.num_layers, inputs.shape[0], self.hidden_size), ctx=inputs.context)
        output, hidden = self.rnn(inputs, hidden)
        return output, hidden[0]