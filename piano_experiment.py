__author__ = 'mike'

import theano
from theano import config

floatX = config.floatX
import numpy as np
from theano import tensor as T
from blocks.bricks.recurrent import SimpleRecurrent, recurrent, BaseRecurrent

from blocks.bricks.cost import CategoricalCrossEntropy, MisclassificationRate, SquaredError
from blocks.initialization import IsotropicGaussian, Constant
from blocks.algorithms import GradientDescent, Adam
from blocks.bricks import Tanh, MLP, Linear
from blocks.main_loop import MainLoop
from fuel.streams import DataStream
from fuel.schemes import SequentialScheme
from blocks.model import Model
from midi import MidiSequence
from fuel.transformers import Padding, Mapping
from blocks.extensions import Timing, Printing, FinishAfter
from blocks.extensions.monitoring import TrainingDataMonitoring, DataStreamMonitoring
from blocks.monitoring import aggregation
from blocks.extensions.plot import Plot

from blocks.graph import ComputationGraph


def test_value(variable, test_val):
    variable.tag.test_value = test_val
    return variable


x = T.tensor3('features')
x_mask = T.matrix('features_mask')
y = T.tensor3('targets')
y_mask = T.matrix('targets_mask')

x = test_value(x, np.ones((15, 10, 93), dtype=floatX))
y = test_value(y, np.ones((15, 10, 93), dtype=floatX))
x_mask = test_value(x_mask, np.ones((15, 10), dtype=floatX))
y_mask = test_value(y_mask, np.ones((15, 10), dtype=floatX))

# h = rnn_layer.apply(x)

class SimpleRNN(BaseRecurrent):
    def __init__(self, dims, **kwargs):
        super(SimpleRNN, self).__init__(**kwargs)
        self.dim = dims[0]
        self.dims = dims
        # self.in_layer = MLP(activations=[Linear(dims[0], dims[1])], dims=[dims[0], dims[1]],
        # weights_init=IsotropicGaussian(0.01),
        #                     biases_init=Constant(0.0),
        #                     name="in_layer")
        self.in_layer = Linear(dims[0], dims[1],
                               weights_init=IsotropicGaussian(0.01),
                               biases_init=Constant(0.0),
                               name="in_layer")

        self.rnn_layer = SimpleRecurrent(dim=dims[1], activation=Tanh(),
                                         weights_init=IsotropicGaussian(0.01),
                                         biases_init=Constant(0.0),
                                         name="rnn_layer")

        self.out_layer = MLP(activations=[Tanh()], dims=[dims[1], dims[2]],
                             weights_init=IsotropicGaussian(0.01),
                             biases_init=Constant(0.0),
                             name="out_layer")

        self.children = [self.in_layer, self.rnn_layer, self.out_layer]

    @recurrent(sequences=['inputs'], contexts=[], states=['hidden_state'],
               outputs=['hidden_state', 'output'])
    def apply(self, inputs, hidden_state=None, output=None):
        h_in = self.in_layer.apply(inputs)
        hidden_state = self.rnn_layer.apply(inputs=h_in, states=hidden_state, iterate=False)
        output = self.out_layer.apply(hidden_state)
        return hidden_state, output

    def get_dim(self, name):
        dims = dict(zip(['inputs', 'hidden_state', 'output'], self.dims))
        return dims.get(name, None) or super(SimpleRNN, self).get_dim(name)


train = MidiSequence('nottingham')
test = MidiSequence('nottingham', which_set='test')


def get_datastream(dataset):
    dataset = DataStream(
        dataset,
        iteration_scheme=SequentialScheme(
            dataset.num_examples, 400
        ),
    )
    dataset = Padding(dataset)

    def _transpose(data):
        return tuple(np.rollaxis(array, 1, 0) for array in data)

    dataset = Mapping(dataset, _transpose)
    return dataset


train = get_datastream(train)
test = get_datastream(test)

srnn = SimpleRNN(dims=[93, 200, 93])
srnn.initialize()

probs = srnn.apply(inputs=x)[1]
probs = (probs.dimshuffle(2, 0, 1) * x_mask).dimshuffle(1, 2, 0)
target = (y.dimshuffle(2, 0, 1) * y_mask).dimshuffle(1, 2, 0)
cost = SquaredError().apply(target, probs)
# error_rate = MisclassificationRate().apply(target, probs)
model = Model(cost)
cg = ComputationGraph([cost])
cost.name = "final_cost"

algorithm = GradientDescent(step_rule=Adam(), cost=cost, params=cg.parameters)

extensions = [
              FinishAfter(after_n_epochs=1000),
              DataStreamMonitoring(
                  [cost],
                  test,
                  prefix="test"),
              TrainingDataMonitoring(
                  [cost,
                   aggregation.mean(algorithm.total_gradient_norm)],
                  prefix="train",
                  after_epoch=True),
              Timing(),
              Printing()]
bokeh = True
if bokeh:
    extensions.append(Plot(
        'Testing blocks',
        channels=[
            ['test_final_cost', 'train_final_cost'],
        ]))

main_loop = MainLoop(algorithm=algorithm,
                     data_stream=train,
                     model=model,
                     extensions=extensions
                     )
main_loop.run()