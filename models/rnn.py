__author__ = 'mike'

import sys

from blocks.bricks.recurrent import BaseRecurrent, LSTM, recurrent, Initializable, SimpleRecurrent
from blocks.bricks import Linear, MLP, Tanh, Sigmoid
from blocks.initialization import IsotropicGaussian, Constant
import theano
import numpy as np


sys.setrecursionlimit(10000)

from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

rng = RandomStreams(seed=np.random.randint(1 << 30))

floatX = theano.config.floatX


class Rnn(Initializable, BaseRecurrent):
    def __init__(self, dims=(88, 100, 100), **kwargs):
        super(Rnn, self).__init__(**kwargs)
        self.dims = dims

        self.input_transform = Linear(input_dim=dims[0], output_dim=dims[1],
                                      weights_init=IsotropicGaussian(0.01),
                                      # biases_init=Constant(0.0),
                                      use_bias=False,
                                      name="input_transfrom")

        self.gru_layer = SimpleRecurrent(dim=dims[1], activation=Tanh(),
                                         weights_init=IsotropicGaussian(0.01),
                                         biases_init=Constant(0.0),
                                         use_bias=True,
                                         name="gru_rnn_layer")

        # TODO: find a way to automatically set the output dim in case of lstm vs normal rnn
        self.linear_trans = Linear(input_dim=dims[1], output_dim=dims[2] * 4,
                                   weights_init=IsotropicGaussian(0.01),
                                   biases_init=Constant(0.0),
                                   use_bias=False,
                                   name="h2h_transform")

        self.lstm_layer = LSTM(dim=dims[2], activation=Tanh(),
                               weights_init=IsotropicGaussian(0.01),
                               biases_init=Constant(0.0),
                               use_bias=True,
                               name="lstm_rnn_layer")

        self.out_transform = MLP(activations=[Sigmoid()], dims=[dims[2], dims[0]],
                                 weights_init=IsotropicGaussian(0.01),
                                 use_bias=True,
                                 biases_init=Constant(0.0),
                                 name="out_layer")

        self.children = [self.input_transform, self.gru_layer, self.linear_trans,
                         self.lstm_layer, self.out_transform]

    # @recurrent(sequences=['inputs', 'input_mask'], contexts=[],
    # states=['gru_state', 'lstm_state', 'lstm_cells'],
    # outputs=['gru_state', 'lstm_state', 'lstm_cells'])
    def rnn_apply(self, inputs, mask=None, gru_state=None, lstm_state=None, lstm_cells=None):
        input_transform = self.input_transform.apply(inputs)
        gru_state = self.gru_layer.apply(
            inputs=input_transform,
            # update_inputs=input_transform,
            # reset_inputs=input_transform,
            states=gru_state,
            mask=mask,
            iterate=False)
        lstm_transform = self.linear_trans.apply(gru_state)
        lstm_state, lstm_cells = self.lstm_layer.apply(inputs=lstm_transform, states=lstm_state,
                                                       cells=lstm_cells,
                                                       mask=mask, iterate=False)
        return gru_state, lstm_state, lstm_cells

    @recurrent(sequences=[], contexts=[],
               states=['inputs', 'gru_state', 'lstm_state', 'lstm_cells'],
               outputs=['inputs', 'gru_state', 'lstm_state', 'lstm_cells'])
    def rnn_generate(self, inputs=None, gru_state=None, lstm_state=None, lstm_cells=None):
        output = self.apply(inputs=inputs,
                            gru_state=gru_state,
                            lstm_state=lstm_state,
                            lstm_cells=lstm_cells,
                            iterate=False)
        return output, gru_state, lstm_state, lstm_cells


    @recurrent(sequences=['inputs', 'mask'], contexts=[],
               states=['gru_state', 'lstm_state', 'lstm_cells'],
               outputs=['output', 'gru_state', 'lstm_state', 'lstm_cells'])
    def apply(self, inputs, mask, gru_state=None, lstm_state=None, lstm_cells=None):
        # input_transform = self.input_transform.apply(inputs)
        # gru_state = self.gru_layer.apply(
        # inputs=input_transform,
        #     mask=mask,
        #     states=gru_state,
        #     iterate=False)
        # lstm_transform = self.linear_trans.apply(gru_state)
        # lstm_state, lstm_cells = self.lstm_layer.apply(inputs=lstm_transform, states=lstm_state,
        #                                                cells=lstm_cells,
        #                                                mask=mask, iterate=False)
        gru_state, lstm_state, lstm_cells = self.rnn_apply(inputs=inputs,
                                                           mask=mask,
                                                           gru_state=gru_state,
                                                           lstm_state=lstm_state,
                                                           lstm_cells=lstm_cells)

        output = 1.17 * self.out_transform.apply(lstm_state) * mask[:, None]
        return output, gru_state, lstm_state, lstm_cells


    def get_dim(self, name):
        dims = dict(zip(['outputs', 'gru_state', 'lstm_state'], self.dims))
        dims['lstm_cells'] = dims['lstm_state']
        return dims.get(name, None) or super(Rnn, self).get_dim(name)