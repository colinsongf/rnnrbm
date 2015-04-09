__author__ = 'mike'

import theano
from theano import config

floatX = config.floatX
import numpy as np
from theano import tensor as T
from blocks.bricks.recurrent import SimpleRecurrent, recurrent, Tanh, BaseRecurrent

from blocks.bricks.cost import CategoricalCrossEntropy
from blocks.initialization import IsotropicGaussian, Constant
from blocks.algorithms import GradientDescent, Adam
from blocks.main_loop import MainLoop
from fuel.streams import DataStream
from fuel.schemes import SequentialScheme
from blocks.model import Model
from midi import MidiSequence
from fuel.transformers import Padding, Mapping
from blocks.extensions import Timing, Printing
from blocks.graph import ComputationGraph


x = T.tensor3('features')
x_mask = T.tensor3('features_mask')
y = T.tensor3('targets')
y_mask = T.tensor3('targets_mask')

# h = rnn_layer.apply(x)

class SimpleRNN(BaseRecurrent):
    def __init__(self, dim, **kwargs):
        super(SimpleRNN, self).__init__(**kwargs)
        self.dim = dim
        self.rnn_layer = SimpleRecurrent(dim=self.dim, activation=Tanh(), weights_init=IsotropicGaussian(0.01),
                                         biases_init=Constant(0.0),
                                         name="rnn_layer")
        self.out_layer = Tanh(name="out_layer")
        self.children = [self.rnn_layer, self.out_layer]

    @recurrent(sequences=['inputs'], contexts=[], states=['hidden_state'], outputs=['hidden_state', 'output'])
    def apply(self, inputs, hidden_state=None, output=None):
        hidden_state = self.rnn_layer.apply(inputs=inputs, states=hidden_state, iterate=False)
        output = self.out_layer.apply(hidden_state)
        return hidden_state, output

    def get_dim(self, name):
        return (self.dim if name in ['inputs', 'hidden_state', 'output'] \
                    else super(SimpleRNN, self).get_dim(name))


nottingham = MidiSequence('nottingham')
dataset = DataStream(
    nottingham,
    iteration_scheme=SequentialScheme(
        nottingham.num_examples, 100
    ),
)
dataset = Padding(dataset)


def _transpose(data):
    return tuple(np.rollaxis(array, 1, 0) for array in data)


dataset = Mapping(dataset, _transpose)


srnn = SimpleRNN(dim=96)
srnn.initialize()

probs = srnn.apply(inputs=x)[1]
cost = CategoricalCrossEntropy().apply(y, probs)
model = Model(cost)
cg = ComputationGraph(cost)

algorithm = GradientDescent(step_rule=Adam(), cost=cost, params=cg.parameters)
main_loop = MainLoop(algorithm=algorithm,
                     data_stream=dataset,
                     model=model,
                     extensions=[Timing(), Printing()]
                     )
main_loop.run()