from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import mxnet as mx
from mxnet import gluon
from seq2seq.encoder import Encoder
from seq2seq.decoder import Decoder

class Seq2seq(gluon.Block):
    def __init__(self, vocab_size, hidden_size, num_layers, dropout, sample_rate=0.4, **kwargs):
        super(Seq2seq, self).__init__(**kwargs)
        with self.name_scope():
            self.encoder = Encoder(hidden_size, num_layers, dropout)
            self.decoder = Decoder(vocab_size, hidden_size, 1, sample_rate)
            self.hidden_size = hidden_size
            self.num_layers = num_layers

    def forward(self, inputs, targets):
        '''
        `inputs`: (batch, length, dim)
        `targets`: (batch, length)
        '''
        enc_out, enc_hid = self.encoder(inputs)
        output, loss = self.decoder(targets, enc_out, enc_hid)
        return output, loss