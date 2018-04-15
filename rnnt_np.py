import mxnet as mx
import numpy as np

def forward_pass(log_probs, labels, blank):

    T, U, _ = log_probs.shape
    alphas = np.zeros((T, U))

    for t in range(1, T):
        alphas[t, 0] = alphas[t-1, 0] + log_probs[t-1, 0, blank]

    for u in range(1, U):
        alphas[0, u] = alphas[0, u-1] + log_probs[0, u-1, labels[u-1]]
    for t in range(1, T):
        for u in range(1, U):
            no_emit = alphas[t-1, u] + log_probs[t-1, u, blank]
            emit = alphas[t, u-1] + log_probs[t, u-1, labels[u-1]]
            alphas[t, u] = np.logaddexp(emit, no_emit)

    loglike = alphas[T-1, U-1] + log_probs[T-1, U-1, blank]
    return alphas, loglike

def backward_pass(log_probs, labels, blank):

    T, U, _ = log_probs.shape
    betas = np.zeros((T, U))
    betas[T-1, U-1] = log_probs[T-1, U-1, blank]

    for t in reversed(range(T-1)):
        betas[t, U-1] = betas[t+1, U-1] + log_probs[t, U-1, blank]

    for u in reversed(range(U-1)):
        betas[T-1, u] = betas[T-1, u+1] + log_probs[T-1, u, labels[u]]

    for t in reversed(range(T-1)):
        for u in reversed(range(U-1)):
            no_emit = betas[t+1, u] + log_probs[t, u, blank]
            emit = betas[t, u+1] + log_probs[t, u, labels[u]]
            betas[t, u] = np.logaddexp(emit, no_emit)

    return betas, betas[0, 0]

def compute_gradient(log_probs, alphas, betas, labels, blank):
    T, U, _ = log_probs.shape
    grads = np.full(log_probs.shape, -float("inf"))
    log_like = betas[0, 0]

    grads[T-1, U-1, blank] = alphas[T-1, U-1]

    grads[:T-1, :, blank] = alphas[:T-1, :] + betas[1:, :]
    for u, l in enumerate(labels):
        grads[:, u, l] = alphas[:, u] + betas[:, u+1]

    grads = -np.exp(grads - log_like)
    return grads

def transduce(log_probs, labels, blank=0):
    """
    Args:
        acts: 3D array with shape
              [input len, output len + 1, vocab size]
        labels: 1D array with shape [output time steps]
    Returns:
        float: The negative log-likelihood
        3D array: Gradients with respect to the
                    unnormalized input actications
    """
    alphas, ll_forward = forward_pass(log_probs, labels, blank)
    betas, ll_backward = backward_pass(log_probs, labels, blank)
    grads = compute_gradient(log_probs, alphas, betas, labels, blank)
    return -ll_forward, grads

def transduce_batch(log_probs, labels, flen, glen, blank=0):
    grads = np.zeros_like(log_probs)
    costs = []
    # TODO parallel loop
    for b in range(log_probs.shape[0]):
        t = int(flen[b])
        u = int(glen[b]) + 1
        ll, g = transduce(log_probs[b, :t, :u, :], labels[b, :u-1], blank)
        grads[b, :t, :u, :] = g
        costs.append(ll)
    return costs, grads

class RNNTransducer(mx.operator.CustomOp):
    """The implementation of RNN Transducer loss functions.

    To make it usable for real-world cases, this class has two policies below.
    1. This class computes forward and backward variables in the log domain.
    2. This class do not apply the softmax function to inputs, since the gradient calculation will be easily overflow. 

    """
    def __init__(self, blank):
        self.blank = blank

    def forward(self, is_train, req, in_data, out_data, aux):
        '''
        `ytu`: am & pm joint probability, layout 'BTUV'
        `y`: label sequence (blank, y1, ..., yU), layout 'BU'
        `flen`: acoustic model outputs sequence true length <= T
        `glen`: label sequence length <= U
        '''
        ytu, y, flen, glen = in_data
        logp = ytu.log()

        loss, grad = transduce_batch(logp.asnumpy(), y.asnumpy().astype(np.int32), flen.asnumpy(), glen.asnumpy(), self.blank)
        self.saved_tensors = mx.nd.array(grad, ctx=logp.context),

        self.assign(out_data[0], req[0], mx.nd.array(loss, ctx=logp.context))

    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
        grad,  = self.saved_tensors
        self.assign(in_grad[0], req[0], grad)


@mx.operator.register('Transducer')
class RNNTransducerProp(mx.operator.CustomOpProp):
    def __init__(self, blank=0):
        super(RNNTransducerProp, self).__init__()
        self.blank = int(blank)

    def list_arguments(self):
        return ['ytu', 'label', 'flen', 'glen']

    def list_outputs(self):
        return ['output']

    def infer_shape(self, in_shapes):
        return in_shapes, ((in_shapes[1][0],),), ()

    def create_operator(self, ctx, shapes, dtypes):
        return RNNTransducer(self.blank)
    
class RNNTLoss(mx.gluon.loss.Loss):
    def __init__(self, blank=0, weight=None, **kwargs):
        batch_axis = 0
        self.blank = blank
        super(RNNTLoss, self).__init__(weight, batch_axis, **kwargs)
    
    def hybrid_forward(self, F, ytu, label, flen, glen):
        loss = F.Custom(ytu, label, flen, glen, blank=self.blank, op_type='Transducer')
        return loss

if __name__ == '__main__':
    T = 400; U = 300; B = 8; V = 50
    ctx = mx.cpu()
    def joint_test():
        ytu = mx.nd.softmax(mx.nd.random_uniform(-10, 10, shape=(B, T, U+1, V), ctx=ctx, dtype=np.float32), axis=3)
        y = mx.nd.random_uniform(1, V, shape=(B, U), ctx=ctx).astype('i')
        flen = mx.nd.full(B, T, ctx=ctx, dtype='i')
        glen = mx.nd.full(B, U, ctx=ctx, dtype='i')
        ytu.attach_grad()
        with mx.autograd.record():
            loss = mx.nd.Custom(ytu, y, flen, glen, op_type='Transducer')
        loss.backward()
        print(ytu.grad)
        print(loss)
    def seperate_test():
        f = mx.nd.random_uniform(-10, 10, shape=(B, T, V), ctx=ctx, dtype=np.float32)
        g = mx.nd.random_uniform(-10, 10, shape=(B, U+1, V), ctx=ctx, dtype=np.float32)
        y = mx.nd.random_uniform(1, V, shape=(B, U), ctx=ctx).astype('i')
        flen = mx.nd.full(B, T, ctx=ctx, dtype='i')
        glen = mx.nd.full(B, U, ctx=ctx, dtype='i')
        f.attach_grad()
        g.attach_grad()
        with mx.autograd.record():
            f1 = mx.nd.expand_dims(f, axis=2)
            g1 = mx.nd.expand_dims(g, axis=1)
            ytu = mx.nd.softmax(f1 + g1, axis=3)
            loss = mx.nd.Custom(ytu, y, flen, glen, op_type='Transducer', blank=0)
        loss.backward()
        print(f.grad)
        print(g.grad)
        print(loss)
    def loss_test():
        ytu = mx.nd.softmax(mx.nd.random_uniform(-10, 10, shape=(B, T, U+1, V), ctx=ctx, dtype=np.float32), axis=3)
        y = mx.nd.random_uniform(1, V, shape=(B, U), ctx=ctx).astype('i')
        flen = mx.nd.full(B, T, ctx=ctx, dtype='i')
        glen = mx.nd.full(B, U, ctx=ctx, dtype='i')
        ytu.attach_grad()
        criterion = RNNTLoss()
        with mx.autograd.record():
            loss = criterion(ytu, y, flen, glen)
        loss.backward()
        print(ytu.grad)
        print(loss)
    # seperate_test()
    loss_test()