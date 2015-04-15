__author__ = 'mike'

import theano
from theano import config

floatX = config.floatX
import numpy as np
from theano import tensor as T
from blocks.bricks.recurrent import SimpleRecurrent, recurrent, BaseRecurrent, LSTM

from utils import MismulitclassificationRate, MeanSquare, ParametricRectifier, MismulitmistakeRate, NanRectify
from blocks.bricks.cost import CategoricalCrossEntropy, SquaredError
from blocks.initialization import IsotropicGaussian, Constant, Identity
from blocks.algorithms import GradientDescent, Adam, Scale, CompositeRule, StepClipping, RemoveNotFinite
from blocks.bricks import Tanh, MLP, Linear, Sigmoid, Rectifier, WEIGHT, BIAS
from blocks import bricks
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
from blocks.extensions.saveload import Checkpoint

from blocks.filter import VariableFilter

from blocks.graph import ComputationGraph
from blocks.config_parser import config

config.recursion_limit = 100000


def test_value(variable, test_val):
    variable.tag.test_value = test_val
    return variable


np.set_printoptions(threshold='nan')

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

        self.in_layer = Linear(input_dim=dims[0], output_dim=dims[1],
                               weights_init=IsotropicGaussian(0.001),
                               use_bias=False,
                               # biases_init=Constant(0.0),
                               name="in_layer")

        self.rnn_layer = SimpleRecurrent(dim=dims[1], activation=NanRectify(),
                                         weights_init=Identity(0.01),
                                         biases_init=Constant(0.0),
                                         use_bias=True,
                                         name="rnn_layer")

        # self.in_layer = Linear(input_dim=dims[0], output_dim=dims[1] * 4,
        # weights_init=IsotropicGaussian(0.1),
        # biases_init=Constant(0.0),
        # name="in_layer")
        #
        # self.rnn_layer = LSTM(dim=dims[1], activation=Tanh(),
        # weights_init=IsotropicGaussian(0.02),
        # biases_init=Constant(0.0),
        # use_bias=True,
        # name="rnn_layer")

        self.out_layer = MLP(activations=[Sigmoid()], dims=[dims[1], dims[2]],
                             weights_init=IsotropicGaussian(0.001),
                             use_bias=False,
                             # biases_init=Constant(0.0),
                             name="out_layer")

        self.children = [self.in_layer, self.rnn_layer, self.out_layer]

    @recurrent(sequences=['inputs', 'input_mask'], contexts=[], states=['hidden_state'],
               outputs=['hidden_state', 'output'])
    def apply(self, inputs, input_mask, hidden_state=None, cells=None, output=None):
        h_in = self.in_layer.apply(inputs)
        # hidden_state, cells = self.rnn_layer.apply(inputs=h_in, states=hidden_state, cells=cells, mask=input_mask,
        # iterate=False)
        hidden_state = self.rnn_layer.apply(inputs=h_in, states=hidden_state, mask=input_mask, iterate=False)
        output = self.out_layer.apply(hidden_state)
        # return hidden_state, cells, output
        return hidden_state, output

    def get_dim(self, name):
        # dims = dict(zip(['inputs', 'hidden_state', 'output'], self.dims))
        dims = dict(zip(['inputs', 'hidden_state', 'output'], self.dims))
        dims['cells'] = dims['hidden_state']
        return dims.get(name, None) or super(SimpleRNN, self).get_dim(name)


train = MidiSequence('nottingham')
test = MidiSequence('nottingham', which_set='test')


def get_datastream(dataset):
    dataset = DataStream(
        dataset,
        iteration_scheme=SequentialScheme(
            dataset.num_examples, 512
        ),
    )
    dataset = Padding(dataset)

    def _transpose(data):
        return tuple(np.rollaxis(array, 1, 0) for array in data)

    dataset = Mapping(dataset, _transpose)
    return dataset


train = get_datastream(train)
test = get_datastream(test)

srnn = SimpleRNN(dims=[93, 128, 93])
srnn.initialize()

hidden_states, probs = srnn.apply(inputs=x, input_mask=x_mask)

debug_val = T.sum(T.isnan(hidden_states))
debug_val.name = 'hidden_states_NANS'

# hidden_states = T.sum(T.isnan(hidden_states))
hidden_states = hidden_states[-1].mean()
hidden_states.name = 'hidden_states'

target = y  # - T.neq(y, 1)
# probs = (probs.dimshuffle(2, 0, 1) * x_mask).dimshuffle(1, 2, 0)
# target = (y.dimshuffle(2, 0, 1) * y_mask).dimshuffle(1, 2, 0)
cost = MeanSquare().apply(target, probs, y_mask)
# cost = SquaredError().apply(target, probs)
error_rate = MismulitclassificationRate().apply(target, probs, y_mask)
error_rate.name = "error on note as a whole"
mistake_rate = MismulitmistakeRate().apply(target, probs, y_mask)
mistake_rate.name = "single error within note"
model = Model(cost)
cg = ComputationGraph([cost])

## l2/l1 regularization
reg = 0.1
params = VariableFilter(roles=[WEIGHT, BIAS])(cg.variables)
for p in params:
    cost += reg * abs(p).sum()
    # cost += reg * (p ** 2).sum()
cost.name = "final_cost"


## Training algorithm
step_rule = CompositeRule([RemoveNotFinite(), StepClipping(20.0), Adam(learning_rate=0.001), StepClipping(6.0),
                           RemoveNotFinite()])  # Scale(0.01)
algorithm = GradientDescent(step_rule=step_rule, cost=cost, params=cg.parameters)

extensions = [FinishAfter(after_n_epochs=5000),
              DataStreamMonitoring(
                  [cost, error_rate, mistake_rate],
                  data_stream=test,
                  prefix="test"),
              TrainingDataMonitoring(
                  [cost, hidden_states, debug_val,
                   aggregation.mean(algorithm.total_gradient_norm)],
                  prefix="train",
                  after_epoch=True),
              Timing(),
              Printing(),
              Checkpoint('piano_experiment_mainloop')]
bokeh = True
if bokeh:
    extensions.append(Plot(
        'Testing blocks',
        channels=[
            ['train_hidden_states_NANS'],
            ['test_error on note as a whole', 'test_single error within note'],
            ['test_final_cost', 'train_final_cost'],
            ['train_total_gradient_norm'],
        ]))

main_loop = MainLoop(algorithm=algorithm,
                     data_stream=train,
                     model=model,
                     extensions=extensions
                     )
main_loop.run()