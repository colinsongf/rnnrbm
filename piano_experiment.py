__author__ = 'mike'

from theano import config

floatX = config.floatX
import numpy as np
from theano import tensor as T
from blocks.bricks.recurrent import recurrent, BaseRecurrent, LSTM

from models import SimpleRNN
from utils import MismulitclassificationRate, MismulitmistakeRate, NegativeLogLikelihood
from blocks.initialization import IsotropicGaussian, Constant
from blocks.algorithms import GradientDescent, Adam, CompositeRule, StepClipping, RemoveNotFinite
from blocks.bricks import Tanh, MLP, Linear, Sigmoid, WEIGHT, BIAS
from blocks.main_loop import MainLoop
from fuel.streams import DataStream
from fuel.schemes import ShuffledScheme
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

inits = [IsotropicGaussian(0.001), IsotropicGaussian(0.001), IsotropicGaussian(0.001)]

train = MidiSequence('nottingham')
test = MidiSequence('nottingham', which_set='test')


def get_datastream(dataset, batch_size=64):
    dataset = DataStream(
        dataset,
        iteration_scheme=ShuffledScheme(
            dataset.num_examples, batch_size
        ),
    )
    dataset = Padding(dataset)

    def _transpose(data):
        return tuple(np.rollaxis(array, 1, 0) for array in data)

    dataset = Mapping(dataset, _transpose)
    return dataset


train = get_datastream(train)
test = get_datastream(test, batch_size=256)

srnn = SimpleRNN(dims=[93, 512, 93])
srnn.initialize()

hidden_states, cells, probs = srnn.apply(inputs=x, input_mask=x_mask)

debug_val = T.sum(T.isnan(hidden_states))
debug_val.name = 'hidden_states_NANS'

# hidden_states = T.sum(T.isnan(hidden_states))
hidden_states = hidden_states[-1].mean()
hidden_states.name = 'hidden_states_mean'

target = y  # - T.neq(y, 1)
# probs = (probs.dimshuffle(2, 0, 1) * x_mask).dimshuffle(1, 2, 0)
# target = (y.dimshuffle(2, 0, 1) * y_mask).dimshuffle(1, 2, 0)
cost = NegativeLogLikelihood().apply(target, probs, y_mask)
# cost = SquaredError().apply(target, probs)
error_rate = MismulitclassificationRate().apply(target, probs, y_mask)
error_rate.name = "error on note as a whole"
mistake_rate = MismulitmistakeRate().apply(target, probs, y_mask)
mistake_rate.name = "single error within note"
model = Model(cost)
cg = ComputationGraph([cost])

## l2/l1 regularization
reg = 0.000005
params = VariableFilter(roles=[WEIGHT, BIAS])(cg.variables)
param_nans = 0
for i, p in enumerate(params):
    # cost += reg * abs(p).sum()
    cost += reg * (p ** 2).sum()
    param_nans += T.isnan(p).sum()
    name = params[i].name
    params[i] = params[i].mean()
    params[i].name = name + str(i)
cost.name = "final_cost"

param_nans.name = 'params_nans'

## Training algorithm
step_rule = CompositeRule([RemoveNotFinite(), StepClipping(20.0), Adam(learning_rate=.001), StepClipping(3.0),
                           RemoveNotFinite()])  # Scale(0.01)
algorithm = GradientDescent(step_rule=step_rule, cost=cost, params=cg.parameters)

extensions = [FinishAfter(after_n_epochs=5000),
              DataStreamMonitoring(
                  [cost, error_rate, mistake_rate],
                  data_stream=test,
                  prefix="test"),
              TrainingDataMonitoring(
                  [cost,  # hidden_states, debug_val, param_nans,
                   aggregation.mean(algorithm.total_gradient_norm)] + params,
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
            ['train_hidden_states_NANS', 'train_params_nans'],
            ['test_error on note as a whole', 'test_single error within note'],
            ['test_final_cost', 'train_final_cost'],
            ['train_total_gradient_norm'],
            ['train_' + param.name for param in params],
        ]))

main_loop = MainLoop(algorithm=algorithm,
                     data_stream=train,
                     model=model,
                     extensions=extensions
                     )
main_loop.run()