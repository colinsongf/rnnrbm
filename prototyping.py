__author__ = 'mike'

## WIP not to be used

import sys

from blocks.bricks.recurrent import BaseRecurrent, LSTM, recurrent
from blocks.bricks import Linear, Tanh, Initializable
from blocks.initialization import IsotropicGaussian, Constant
import theano
import numpy as np
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

rng = RandomStreams(seed=np.random.randint(1 << 30))
floatX = theano.config.floatX
sys.setrecursionlimit(10000)
multiplier = 4


class ModularRnn(BaseRecurrent, Initializable):
    def __init__(self, dims, **kwargs):
        super(ModularRnn, self).__init__(**kwargs)

        self.layers = []
        self.tranforms = []
        for i, (dim1, dim2) in enumerate(zip(dims[:-1], dims[1:])):
            self.layers.append(
                LSTM(dim=dim1, activation=Tanh(),
                     weights_init=IsotropicGaussian(0.01),
                     biases_init=Constant(0.0),
                     use_bias=True,
                     name="rnn_layer%s" % i)
            )
            self.tranforms.append(
                Linear(input_dim=dim1, output_dim=dim2 * multiplier,
                       weights_init=IsotropicGaussian(0.01),
                       biases_init=Constant(0.0),
                       use_bias=False,
                       name="linea_transform%s" % i)
            )

    def create_apply_func(self):
        hidden_state_kwargs = {"hidden_state%s" % i: None for i in range(len(self.layers))}
        hidden_state_kwargs.update({"cells%s" % i: None for i in range(len(self.layers))})

        @recurrent(sequences=['inputs', 'mask'], contexts=[], states=['hidden_state', 'cells'],
                   outputs=['hidden_state', 'cells', 'output'])
        def apply(inputs, mask):
            hidden_states = []
            for i, (layer, transform) in enumerate(zip(self.layers, self.tranforms)):
                hidden_state = layer.apply(inputs=inputs, states=prev_hidden_state, cells=prev_cells, mask=mask,
                                           iterate=False)
                hidden_states.append(hidden_state)
                inputs = hidden_state

            return [outputs] + hidden_states

        setattr(self, 'apply', apply)
