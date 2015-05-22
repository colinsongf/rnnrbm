__author__ = 'mike'

import sys
import time

from blocks.bricks.recurrent import BaseRecurrent, LSTM, recurrent
from blocks.bricks import Linear, MLP, Tanh, Sigmoid, lazy, Initializable
from blocks.initialization import IsotropicGaussian, Constant
from blocks.utils import shared_floatx_nans
from blocks.roles import add_role, WEIGHT, BIAS, has_roles
import theano
import theano.tensor as T
import numpy as np
from blocks.graph import ComputationGraph

from utils import test_value


sys.setrecursionlimit(10000)

from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

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
    def apply(self, inputs, input_mask, hidden_state=None, cells=None):
        h_in = self.in_layer.apply(inputs)
        hidden_state, cells = self.rnn_layer.apply(inputs=h_in, states=hidden_state, cells=cells, mask=input_mask,
                                                   iterate=False)
        # hidden_state = self.rnn_layer.apply(inputs=h_in, states=hidden_state, mask=input_mask, iterate=False)
        output = 1.17 * self.out_layer.apply(hidden_state)
        return hidden_state, cells, output
        # return hidden_state, output


    def get_dim(self, name):
        dims = dict(zip(['inputs', 'hidden_state', 'output'], self.dims))
        dims['cells'] = dims['hidden_state']
        return dims.get(name, None) or super(SimpleRNN, self).get_dim(name)


class Rbm(Initializable, BaseRecurrent):
    @lazy(allocation=['visible_dim', 'hidden_dim'])
    def __init__(self, visible_dim, hidden_dim, activation=Sigmoid(), **kwargs):
        super(Rbm, self).__init__(**kwargs)
        self.hidden_dim = hidden_dim
        self.visible_dim = visible_dim
        self.activation = activation
        self.children = [activation]

    @property
    def W(self):
        return self.params[0]

    @W.setter
    def W(self, value):
        self.params[0] = value

    @property
    def bv(self):
        return self.params[1]

    @property
    def bh(self):
        return self.params[2]

    def _allocate(self):
        W = shared_floatx_nans((self.visible_dim, self.hidden_dim), name='Wrbm')
        add_role(W, WEIGHT)
        self.params.append(W)
        self.add_auxiliary_variable(W.norm(2), name='W_norm')
        bv = shared_floatx_nans((self.visible_dim,), name='bv')
        add_role(bv, BIAS)
        self.params.append(bv)
        self.add_auxiliary_variable(bv.norm(2), name='bv_norm')
        bh = shared_floatx_nans((self.hidden_dim,), name='bh')
        add_role(bh, BIAS)
        self.params.append(bh)
        self.add_auxiliary_variable(bh.norm(2), name='bh_norm')


    def _initialize(self):
        for param in self.params:
            if has_roles(param, [WEIGHT]):
                self.weights_init.initialize(param, self.rng)
            elif has_roles(param, [BIAS]):
                self.biases_init.initialize(param, self.rng)

    @recurrent(sequences=[], states=['visible'], outputs=['mean_visible', 'visible'], contexts=['bv', 'bh', 'W'])
    def apply(self, visible=None, bv=None, bh=None):

        mean_hidden = self.activation.apply(T.dot(visible, self.W) + bh)
        h = rng.binomial(size=mean_hidden.shape, n=1, p=mean_hidden,
                         dtype=floatX)
        mean_visible = self.activation.apply(T.dot(h, self.W.T) + bv)
        v = rng.binomial(size=mean_visible.shape, n=1, p=mean_visible,
                         dtype=floatX)
        cg = ComputationGraph(v)
        return [mean_visible, v], cg.updates

    def free_energy(self, v, bv, bh, mask=None):
        bv = self.bv if bv is None else bv
        bh = self.bh if bh is None else bh
        if mask is not None:
            return -(v * bv * mask[:, :, None]).sum() - T.sum(
                mask[:, :, None] * T.log(1 + (T.exp(T.dot(v, self.W) + bh))))
        else:
            return -(v * bv).sum() - T.sum(T.log(1 + (T.exp(T.dot(v, self.W) + bh))))

    def cost(self, visible, k, batch_size, bv=None, bh=None, mask=None):
        _, v_sample = self.apply(visible=visible, bv=bv, bh=bh, n_steps=k, batch_size=batch_size)
        cost = (self.free_energy(visible, bv, bh, mask=mask) - self.free_energy(v_sample[-1], bv, bh, mask=mask)) / \
               visible.shape[0]
        return cost.astype(floatX), v_sample

    def get_dim(self, name):
        if name == 'visible':
            return self.visible_dim
        return super(Rbm, self).get_dim(name)


class Rnnrbm(BaseRecurrent, Initializable):
    @lazy(allocation=['rnn_dimension', 'visible_dim', 'hidden_dim'])
    def __init__(self, rnn_dimension, visible_dim, hidden_dim, **kwargs):
        super(Rnnrbm, self).__init__(**kwargs)
        self.rnn_dimension = rnn_dimension
        self.visible_dim = visible_dim
        self.hidden_dim = hidden_dim

        # self.in_layer = Linear(input_dim=input_dim, output_dim=rnn_dimension * 4,
        # weights_init=IsotropicGaussian(0.01),
        # biases_init=Constant(0.0),
        # use_bias=False,
        # name="in_layer")

        self.rbm = Rbm(visible_dim=visible_dim, hidden_dim=hidden_dim,
                       activation=Sigmoid(), weights_init=IsotropicGaussian(0.1), biases_init=Constant(0.1),
                       name='rbm2l')

        self.uv = Linear(input_dim=rnn_dimension, output_dim=visible_dim,
                         weights_init=IsotropicGaussian(0.001),
                         biases_init=Constant(0.001),
                         use_bias=True, name='uv')

        self.uh = Linear(input_dim=rnn_dimension, output_dim=hidden_dim,
                         weights_init=IsotropicGaussian(0.001),
                         biases_init=Constant(0.001),
                         use_bias=True, name='uh')

        self.vu = Linear(input_dim=visible_dim, output_dim=rnn_dimension * 4,
                         weights_init=IsotropicGaussian(0.1),
                         biases_init=Constant(0.01),
                         use_bias=True, name='vu')

        self.rnn = LSTM(dim=rnn_dimension, activation=Tanh(),
                        weights_init=IsotropicGaussian(0.01),
                        biases_init=Constant(0.0),
                        use_bias=True,
                        name="rnn_layer")

        # self.out_layer = MLP(activations=[Sigmoid()], dims=[rnn_dimension, visible_dim],
        # weights_init=IsotropicGaussian(0.01),
        # use_bias=True,
        # biases_init=Constant(0.0),
        #                      name="out_layer")
        self.children = [self.rbm, self.uv, self.uh, self.vu, self.rnn]  #, self.out_layer,self.in_layer]

    def initialize(self, pretrained_rbm=None, **kwargs):
        super(Rnnrbm, self).initialize(**kwargs)
        if pretrained_rbm is not None:
            self.rbm.bv.set_value(pretrained_rbm.bv.get_value())
            self.uv.b.set_value(pretrained_rbm.bv.get_value())
            self.rbm.bh.set_value(pretrained_rbm.bh.get_value())
            self.uh.b.set_value(pretrained_rbm.bh.get_value())
            self.rbm.W.set_value(pretrained_rbm.W.get_value())
        self.uv.b.name = 'buv'
        self.uv.W.name = 'Wuv'
        self.uh.W.name = 'Wuh'
        self.uh.b.name = 'buh'
        self.vu.b.name = 'bvu'
        self.vu.W.name = 'Wvu'

    # def _allocate(self):
    # Wrbm = shared_floatx_nans((self.visible_dim, self.hidden_dim), name='Wrbm')
    # add_role(Wrbm, WEIGHT)
    # self.params.append(Wrbm)
    # self.add_auxiliary_variable(Wrbm.norm(2), name='Wrbm_norm')
    #
    #
    # def _initialize(self):
    # for param in self.params:
    # if has_roles(param, WEIGHT):
    # self.weights_init.initialize(param, self.rng)
    # elif has_roles(param, BIAS):
    # self.biases_init.initialize(param, self.rng)

    def get_dim(self, name):
        dims = {'visible': self.visible_dim,
                'mask': self.visible_dim,
                'hidden_state': self.rnn_dimension,
                'cells': self.rnn_dimension}
        return dims.get(name, None) or super(Rnnrbm, self).get_dim(name)

    @recurrent(sequences=[], contexts=['visible'], states=['hidden_state', 'cells'],
               outputs=['visible', 'hidden_state', 'cells'])
    def generate(self, visible=None, hidden_state=None, cells=None, rbm_steps=25):
        bv = self.uv.apply(hidden_state)
        bh = self.uh.apply(hidden_state)
        visible_start = T.zeros_like((visible))
        _, visible = self.rbm.apply(visible=visible_start, bv=bv, bh=bh, n_steps=rbm_steps, batch_size=visible.shape[0])
        visible = visible[-1]
        h_in = self.vu.apply(visible)
        hidden_state, cells = self.rnn.apply(inputs=h_in, states=hidden_state, cells=cells,
                                             iterate=False)
        updates = ComputationGraph(hidden_state).updates
        # output = 1.17 * self.out_layer.apply(hidden_state)
        return [visible, hidden_state, cells], updates

    @recurrent(sequences=['visible', 'mask'], contexts=[], states=['hidden_state', 'cells'],
               outputs=['hidden_state', 'cells', 'bv', 'bh'])
    def training_biases(self, visible, mask=None, hidden_state=None, cells=None):
        bv = self.uv.apply(hidden_state)
        bh = self.uh.apply(hidden_state)
        # inputs = rbm.apply(visible=visible, bv=bv, bh=bh, n_steps=25)
        h_in = self.vu.apply(visible)
        hidden_state, cells = self.rnn.apply(inputs=h_in, states=hidden_state, cells=cells, mask=mask,
                                             iterate=False)
        updates = ComputationGraph(hidden_state).updates
        return [hidden_state, cells, bv, bh], updates

    def cost(self, examples, mask, k=10):
        _, _, bv, bh = self.training_biases(visible=examples, mask=mask)
        cost, v_samples = self.rbm.cost(visible=examples, bv=bv, bh=bh, k=k, batch_size=examples.shape[0])
        return cost.astype(floatX), v_samples.astype(floatX)


if __name__ == "__main__":
    visible_dim = 100
    hidden_dim = 200
    # v = np.float32(np.random.randn(10, visible_dim))

    x = T.tensor3('features')
    x_mask = T.matrix('features_mask')
    y = T.tensor3('targets')
    y_mask = T.matrix('targets_mask')

    x = test_value(x, np.ones((15, 10, 93), dtype=floatX))
    y = test_value(y, np.ones((15, 10, 93), dtype=floatX))
    x_mask = test_value(x_mask, np.ones((15, 10), dtype=floatX))
    y_mask = test_value(y_mask, np.ones((15, 10), dtype=floatX))

    rnnrbm = Rnnrbm(93, 256, 93, 300)
    rnnrbm.allocate()
    rnnrbm.initialize()

    # y = rnnrbm.generate(n_steps=5, batch_size=x.shape[1])
    y, m = rnnrbm.cost(x, x_mask)

    print y
    # rbm = Rbm(visible_dim, hidden_dim, weights_init=IsotropicGaussian(0.1), use_bias=False, name='rbm')
    # rbm.allocate()
    # rbm.initialize()
    # # print rbm.initial_state

    # a = T.ones(visible_dim, dtype=floatX) * 0.5
    # b = T.ones(hidden_dim, dtype=floatX) * 0.5
    # y = rbm.apply(visible=x, bv=a, bh=b,
    # n_steps=10, batch_size=10)
    cg = ComputationGraph(y)
    updates = cg.updates
    #
    f = theano.function([x, x_mask], y, updates=updates)
    for i in range(100):
        t = time.time()
        print f(np.ones((15, 10, 93), dtype=floatX), np.ones((15, 10), dtype=floatX))
        print "took: " + str(time.time() - t)
