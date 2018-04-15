import random
import mxnet as mx
from mxnet import gluon
from mxnet.gluon import nn, rnn
from seq2seq.attention import Attention

class Decoder(gluon.Block):
    def __init__(self, vocab_size, hidden_size, num_layers, sample_rate, **kwargs):
        super(Decoder, self).__init__(**kwargs)
        with self.name_scope():
            self.embedding = nn.Embedding(vocab_size, hidden_size)
            self.attention = Attention()
            # self.rnn = rnn.GRU(hidden_size, num_layers, layout='NTC', input_size=hidden_size*2)
            self.rnn = rnn.GRUCell(hidden_size, input_size=hidden_size*2) # NOTE only use one layer cell
            self.fc = nn.Dense(vocab_size - 1) # NOTE output should not predict '<sos>'
            self.vocab_size = vocab_size
            self.loss = gluon.loss.SoftmaxCELoss()
            self.num_layers = num_layers
            self.hidden_size = hidden_size
            self.sample_rate = sample_rate
        
    def forward(self, target, enc_out, enc_hid):
        '''
        `target`: (batch, length)
        `enc_out`: Encoder output, (batch, length, hidden_size)
        `enc_hid`: last hidden state of encoder 
        TODO Truncated BPTT
        '''
        target = target.transpose(axes=(1,0))
        length, batch_size = target.shape
        result = mx.nd.zeros((length-1, batch_size, self.vocab_size-1), ctx=target.context)
        inputs = target[0]
        hidden = enc_hid[-1] # NOTE decoder only use last hidden of encoder
        loss = 0
        # target remove beginning 'sos'
        for i in range(1, length):
            output, hidden = self._step(inputs, hidden, enc_out)
            result[i - 1] = output
            if self.sample_rate > 0 and random.random() < self.sample_rate:
                inputs = output.argmax(axis=1)
            else:
                inputs = target[i]
            loss = loss + self.loss(output, target[i])

        return result.transpose(axes=(1, 0, 2)), loss

    def _step(self, inputs, hidden, enc_out):
        embedded = self.embedding(inputs) # BH
        att = self.attention(hidden, enc_out) # BH
        output, hidden = self.rnn(mx.nd.concat(embedded, att, dim=1), [hidden])
        output = self.fc(output)
        return output, hidden[0] # NOTE for next attention calculation
