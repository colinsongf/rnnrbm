import time

__author__ = 'mike'

import sys

from blocks.bricks.recurrent import BaseRecurrent, LSTM, recurrent
from blocks.bricks import Linear, MLP, Tanh, Sigmoid, lazy, Initializable
from blocks.initialization import IsotropicGaussian, Constant
from blocks.utils import shared_floatx_nans
from blocks.roles import add_role, WEIGHT, BIAS, has_roles
from blocks.graph import ComputationGraph
import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams
import numpy as np

sys.setrecursionlimit(10000)

rng = RandomStreams(seed=np.random.randint(1 << 30))
floatX = theano.config.floatX


class SimpleRNN(BaseRecurrent):
    def __init__(self, dims, **kwargs):
        super(SimpleRNN, self).__init__(**kwargs)
        self.dim = dims[0]
        self.dims = dims[:2] + dims[-1:]

        # self.in_layer = Linear(input_dim=dims[0], output_dim=dims[1],
        # weights_init=IsotropicGaussian(0.01),
        # use_bias=False,
        # # biases_init=Constant(0.0),
        # name="in_layer")

        # self.rnn_layer = SimpleRecurrent(dim=dims[1], activation=NanRectify(),
        # weights_init=Identity(0.5),
        # biases_init=Constant(0.0),
        # use_bias=True,
        # name="rnn_layer")

        self.in_layer = Linear(input_dim=dims[0], output_dim=dims[1] * 4,
                               weights_init=IsotropicGaussian(0.01),
                               biases_init=Constant(0.0),
                               use_bias=False,
                               name="in_layer")

        self.rnn_layer = LSTM(dim=dims[1], activation=Tanh(),
                              weights_init=IsotropicGaussian(0.01),
                              biases_init=Constant(0.0),
                              use_bias=True,
                              name="rnn_layer")

        self.out_layer = MLP(activations=[Sigmoid()], dims=[dims[1], dims[2]],
                             weights_init=IsotropicGaussian(0.01),
                             use_bias=True,
                             biases_init=Constant(0.0),
                             name="out_layer")

        self.children = [self.in_layer, self.rnn_layer, self.out_layer]

    @recurrent(sequences=['inputs', 'input_mask'], contexts=[], states=['hidden_state', 'cells'],
               outputs=['hidden_state', 'cells', 'output'])
    def apply(self, inputs, input_mask, hidden_state=None, cells=None, output=None):
        h_in = self.in_layer.apply(inputs)
        hidden_state, cells = self.rnn_layer.apply(inputs=h_in, states=hidden_state, cells=cells, mask=input_mask,
                                                   iterate=False)
        # hidden_state = self.rnn_layer.apply(inputs=h_in, states=hidden_state, mask=input_mask, iterate=False)
        output = 1.17 * self.out_layer.apply(hidden_state)
        return hidden_state, cells, output
        # return hidden_state, output

    def generate(self, ):
        pass


    def get_dim(self, name):
        dims = dict(zip(['inputs', 'hidden_state', 'output'], self.dims))
        dims['cells'] = dims['hidden_state']
        return dims.get(name, None) or super(SimpleRNN, self).get_dim(name)


class Rbm(Initializable):
    @lazy(allocation=['visible_dim', 'hidden_dim'])
    def __init__(self, dimensions, activation=Sigmoid(), **kwargs):
        super(Rbm, self).__init__(**kwargs)
        self.dimensions = dimensions
        self.activation = activation
        self.children = [activation]
        self.weights = []
    # @property
    # def W(self):
    # return self.params[0]
    #
    # @W.setter
    # def W(self, value):
    # self.params[0] = value

    def _allocate(self):
        for f, t in zip(self.dimensions[:-1], self.dimensions[1:]):
            W = shared_floatx_nans((f, t), name='W')
            add_role(W, WEIGHT)
            self.params.append(W)
            self.weights.append(W)
            self.add_auxiliary_variable(W.norm(2), name='W_norm')

    def _initialize(self):
        for param in self.params:
            if has_roles(param, WEIGHT):
                self.weights_init.initialize(param, self.rng)
            elif has_roles(param, BIAS):
                self.biases_init.initialize(param, self.rng)

    @recurrent(sequences=[], states=['next_state'], outputs=['mean_visible', 'next_state'], contexts=['bv', 'bh', 'W'])
    def apply(self, next_state, bv=None, bh=None):
        mean_hidden,h = [],[]
        for i in range(len(self.dimensions), step=2):
            activation_input = T.dot(next_state, self.W)
            mean_hidden[i] = self.activation.apply(T.dot(next_state, self.W) + bh)
            h[i] = rng.binomial(size=mean_hidden.shape, n=1, p=mean_hidden,
                             dtype=floatX)
        for
            mean_visible = self.activation.apply(T.dot(h, self.W.T) + bv)
        v = rng.binomial(size=mean_visible.shape, n=1, p=mean_visible,
                         dtype=floatX)
        return mean_visible, v

    def free_energy(self, v, bv, bh):
        return -(v * bv).sum() - T.log(1 + T.exp(T.dot(v, self.W) + bh)).sum()

    def cost(self, v, bv, bh, k, batch_size):
        v_sample = self.apply(v, bv=bv, bh=bh, n_steps=k, batch_size=batch_size)
        cost = (self.free_energy(v, bv, bh) - self.free_energy(v_sample, bv, bh)) / v.shape[0]
        return cost

    def get_dim(self, name):
        if name == 'next_state':
            return self.visible_dim
        return super(Rbm, self).get_dim(name)


class Rnnrbm(BaseRecurrent, Initializable):
    @lazy(allocation=['input_dim', 'output_dim', 'visble_dim', 'hidden_dim'])
    def __init__(self, input_dim, output_dim, **kwargs):
        super(Rnnrbm, self).__init__(**kwargs)
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.visble_dim =

    def _allocate(self):
        Wrbm = shared_floatx_nans((self.visible_dim, self.hidden_dim), name='Wrbm')
        add_role(Wrbm, WEIGHT)
        self.params.append(Wrbm)
        self.add_auxiliary_variable(Wrbm.norm(2), name='Wrbm_norm')

        bv = shared_floatx_nans((self.visible_dim, self.hidden_dim), name='bv')
        add_role(bv, BIAS)
        self.params.append(bv)
        self.add_auxiliary_variable(bv.norm(2), name='bv_norm')

        bh = shared_floatx_nans((self.visible_dim, self.hidden_dim), name='bh')
        add_role(bh, BIASaw)
        self.params.append(bh)
        self.add_auxiliary_variable(bh.norm(2), name='bh_norm')

        Wuh = shared_floatx_nans((self.visible_dim, self.hidden_dim), name='Wuh')
        add_role(Wuh, WEIGHT)
        self.params.append(Wuh)
        self.add_auxiliary_variable(Wuh.norm(2), name='Wuh_norm')

        Wuv = shared_floatx_nans((self.visible_dim, self.hidden_dim), name='W')
        add_role(Wuv, WEIGHT)
        self.params.append(Wuv)
        self.add_auxiliary_variable(Wuv.norm(2), name='Wuv_norm')

        Wvu = shared_floatx_nans((self.visible_dim, self.hidden_dim), name='Wvu')
        add_role(Wvu, WEIGHT)
        self.params.append(Wvu)
        self.add_auxiliary_variable(Wvu.norm(2), name='Wvu_norm')

    def _initialize(self):
        for param in self.params:
            if has_roles(param, WEIGHT):
                self.weights_init.initialize(param, self.rng)
            elif has_roles(param, BIAS):
                self.biases_init.initialize(param, self.rng)

    def apply(self, inputs, input_mask, **kwargs):
        pass
        # return output


if __name__ == "__main__":
    visible_dim = 100
    hidden_dim = 200
    # v = np.float32(np.random.randn(10, visible_dim))

    rbm = Rbm(visible_dim, hidden_dim, weights_init=IsotropicGaussian(0.1), use_bias=False, name='rbm')
    rbm.allocate()
    rbm.initialize()
    # print rbm.initial_state
    x = T.matrix('features')
    x.tag.test_value = np.float32(np.random.randn(10, visible_dim))
    a = T.ones(visible_dim, dtype=floatX) * 0.5
    b = T.ones(hidden_dim, dtype=floatX) * 0.5
    y = rbm.apply(next_state=x, bv=a, bh=b,
                  n_steps=10, batch_size=10)
    cg = ComputationGraph(y)
    updates = cg.updates

    f = theano.function([x], y[-1][-1][-1], updates=updates)
    t = time.time()
    print f(np.float32(np.random.randn(10, visible_dim)))
    print "took: " + str(time.time() - t)
    t = time.time()
    print f(np.float32(np.random.randn(10, visible_dim)))
    print "took: " + str(time.time() - t)
