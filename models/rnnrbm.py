__author__ = 'mike'

import sys

from blocks.bricks.base import application
from blocks.bricks.recurrent import BaseRecurrent, recurrent
from blocks.bricks import Linear, Sigmoid, lazy, Initializable
from blocks.initialization import IsotropicGaussian, Constant
import theano
import theano.tensor as T
import numpy as np
from blocks.graph import ComputationGraph

from utils import test_value
from rbm import Rbm
from rnn import Rnn

sys.setrecursionlimit(10000)

from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

rng = RandomStreams(seed=np.random.randint(1 << 30))

floatX = theano.config.floatX


class Rnnrbm(BaseRecurrent, Initializable):
    @lazy(allocation=['visible_dim', 'hidden_dim'])
    def __init__(self, visible_dim, hidden_dim, rnn_dimensions=(128, 128), **kwargs):
        super(Rnnrbm, self).__init__(**kwargs)
        self.rnn_dimensions = rnn_dimensions
        self.visible_dim = visible_dim
        self.hidden_dim = hidden_dim

        # self.in_layer = Linear(input_dim=input_dim, output_dim=rnn_dimension * 4,
        # weights_init=IsotropicGaussian(0.01),
        # biases_init=Constant(0.0),
        # use_bias=False,
        # name="in_layer")

        self.rbm = Rbm(visible_dim=visible_dim, hidden_dim=hidden_dim,
                       activation=Sigmoid(), weights_init=IsotropicGaussian(0.1),
                       biases_init=Constant(0.1),
                       name='rbm')

        self.uv = Linear(input_dim=rnn_dimensions[-1], output_dim=visible_dim,
                         weights_init=IsotropicGaussian(0.0001),
                         biases_init=Constant(0.001),
                         use_bias=True, name='uv')

        self.uh = Linear(input_dim=rnn_dimensions[-1], output_dim=hidden_dim,
                         weights_init=IsotropicGaussian(0.0001),
                         biases_init=Constant(0.001),
                         use_bias=True, name='uh')

        self.rnn = Rnn([visible_dim] + list(rnn_dimensions), name='rnn')

        self.children = [self.rbm, self.uv, self.uh, self.rnn] + self.rnn.children._items


    def initialize(self, pretrained_rbm=None, pretrained_rnn=None, **kwargs):
        super(Rnnrbm, self).initialize(**kwargs)
        # if pretrained_rbm is not None:
        # self.rbm.bv.set_value(pretrained_rbm.bv.get_value())
        # self.uv.b.set_value(pretrained_rbm.bv.get_value())
        #     self.rbm.bh.set_value(pretrained_rbm.bh.get_value())
        #     self.uh.b.set_value(pretrained_rbm.bh.get_value())
        #     self.rbm.W.set_value(pretrained_rbm.W.get_value())
        # if pretrained_rnn is not None:
        #     for param, trained_param in zip(itertools.chain(*[child.params for child in self.rnn.children]),
        #                                     itertools.chain(*[child.params for child in pretrained_rnn.children])):
        #         param.set_value(trained_param.get_value())
        self.uv.b.name = 'buv'
        self.uv.W.name = 'Wuv'
        self.uh.W.name = 'Wuh'
        self.uh.b.name = 'buh'
        self.rbm.bh.name = 'buh'
        self.rbm.bv.name = 'buv'
        # self.rnn.input_transform.b.name = 'bvu'
        # self.rnn.input_transform.W.name = 'Wvu'

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
                'gru_state': self.rnn_dimensions[0],
                'lstm_state': self.rnn_dimensions[1],
                'lstm_cells': self.rnn_dimensions[1]
                }
        return dims.get(name, None) or super(Rnnrbm, self).get_dim(name)

    @recurrent(sequences=[], contexts=['visible'], states=['gru_state', 'lstm_state', 'lstm_cells'],
               outputs=['visible', 'gru_state', 'lstm_state', 'lstm_cells'])
    def generate(self, visible=None, gru_state=None, lstm_state=None, lstm_cells=None, rbm_steps=25):
        bv = self.uv.apply(lstm_state)
        bh = self.uh.apply(lstm_state)
        visible_start = rng.binomial(size=visible.shape, n=1, p=bv, dtype=floatX)
        _, visible = self.rbm.apply(visible=visible_start, bv=bv, bh=bh, n_steps=rbm_steps,
                                    batch_size=visible.shape[0])
        visible = visible[-1]
        gru_state, lstm_state, lstm_cells = self.rnn.rnn_apply(inputs=visible, gru_state=gru_state,
                                                               lstm_state=lstm_state,
                                                               lstm_cells=lstm_cells, )  # iterate=False)
        # input_transform = self.vu.apply(visible)
        # gru_state = self.gru_layer.apply(inputs=input_transform, states=gru_state,
        # iterate=False)
        # lstm_state, lstm_cells = self.lstm_layer.apply(inputs=gru_state, states=lstm_state, cells=lstm_state,
        # iterate=False)


        updates = ComputationGraph(lstm_state).updates
        # output = 1.17 * self.out_layer.apply(hidden_state)
        return [visible, gru_state, lstm_state, lstm_cells], updates

    @recurrent(sequences=['visible', 'mask'], contexts=[],
               states=['gru_state', 'lstm_state', 'lstm_cells'],
               outputs=['gru_state', 'lstm_state', 'lstm_cells', 'bv', 'bh'])
    def training_biases(self, visible, mask=None, gru_state=None, lstm_state=None, lstm_cells=None):
        bv = self.uv.apply(lstm_state)
        bh = self.uh.apply(lstm_state)
        # inputs = rbm.apply(visible=visible, bv=bv, bh=bh, n_steps=25)
        # input_transform = self.vu.apply(visible)
        # gru_state = self.gru_layer.apply(inputs=input_transform, states=gru_state,
        # iterate=False)
        # lstm_state, lstm_cells = self.lstm_layer.apply(inputs=gru_state, states=lstm_state,
        #                                                cells=lstm_state,
        #                                                iterate=False)
        gru_state, lstm_state, lstm_cells = self.rnn.rnn_apply(visible, mask=mask, gru_state=gru_state,
                                                               lstm_state=lstm_state,
                                                               lstm_cells=lstm_cells, )  #iterate=False)
        updates = ComputationGraph(lstm_state).updates
        return [gru_state, lstm_state, lstm_cells, bv, bh], updates

    def cost(self, examples, mask, k=10):
        _, _, _, bv, bh = self.training_biases(visible=examples, mask=mask)
        cost, v_samples = self.rbm.cost(visible=examples, bv=bv, bh=bh, k=k,
                                        batch_size=examples.shape[0])
        return cost.astype(floatX), v_samples.astype(floatX)

    # @application(inputs=['visible', 'k', 'batch_size', 'mask'], outputs=['vsample', 'cost'])
    def rbm_pretrain_cost(self, visible, k, batch_size, mask=None):
        return self.rbm.cost(visible=visible, k=k, batch_size=batch_size, mask=mask)

    # @application(inputs=['x', 'x_mask'], outputs=['output_'])
    def rnn_pretrain_pred(self, x, x_mask):
        return self.rnn.apply(inputs=x, mask=x_mask)


if __name__ == "__main__":
    # v = np.float32(np.random.randn(10, visible_dim))

    x = T.tensor3('features')
    x_mask = T.matrix('features_mask')
    y = T.tensor3('targets')
    y_mask = T.matrix('targets_mask')

    x = test_value(x, np.ones((15, 10, 88), dtype=floatX))
    y = test_value(y, np.ones((15, 10, 88), dtype=floatX))
    x_mask = test_value(x_mask, np.ones((15, 10), dtype=floatX))
    y_mask = test_value(y_mask, np.ones((15, 10), dtype=floatX))

    rnnrbm = Rnnrbm(88, 256, rnn_dimensions=(100, 100))
    rnnrbm.allocate()
    rnnrbm.initialize()

    # # y = rnnrbm.generate(n_steps=5, batch_size=x.shape[1])
    # # rbm = Rbm(visible_dim, hidden_dim, weights_init=IsotropicGaussian(0.1), use_bias=False, name='rbm')
    # # rbm.allocate()
    # # rbm.initialize()
    # # # print rbm.initial_state
    #
    # # a = T.ones(visible_dim, dtype=floatX) * 0.5
    # # b = T.ones(hidden_dim, dtype=floatX) * 0.5
    # # y = rbm.apply(visible=x, bv=a, bh=b,
    # # n_steps=10, batch_size=10)
    # cg = ComputationGraph(y)
    # updates = cg.updates
    # #
    # f = theano.function([x, x_mask], y, updates=updates)
    # for i in range(100):
    # t = time.time()
    # print f(np.ones((15, 10, 93), dtype=floatX), np.ones((15, 10), dtype=floatX))
    #     print "took: " + str(time.time() - t)
