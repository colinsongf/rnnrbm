__author__ = 'mike'

from theano import config

floatX = config.floatX
import numpy as np
from theano import tensor as T

from models import Rnnrbm, SimpleRNN
from blocks.initialization import IsotropicGaussian
from blocks.algorithms import GradientDescent, Adam, CompositeRule, StepClipping, RemoveNotFinite
from blocks.main_loop import MainLoop
from fuel.streams import DataStream
from fuel.schemes import ShuffledScheme
from blocks.model import Model
from midi import MidiSequence, MidiSequence2
from fuel.transformers import Padding, Mapping
from blocks.extensions import Timing, Printing, FinishAfter
from blocks.extensions.monitoring import TrainingDataMonitoring, DataStreamMonitoring
from blocks.monitoring import aggregation
from blocks.extensions.plot import Plot
from blocks.extensions.saveload import Checkpoint
from picklable_itertools.extras import equizip

from blocks.graph import ComputationGraph
from utils import test_value, MismulitclassificationRate, MismulitmistakeRate, NegativeLogLikelihood

import os
from datetime import datetime

# from blocks.config_parser import config
#
# config.recursion_limit = 100000

np.set_printoptions(threshold='nan')

x = T.tensor3('features')
x_mask = T.matrix('features_mask')
y = T.tensor3('targets')
y_mask = T.matrix('targets_mask')

x = test_value(x, np.ones((10, 15, 93), dtype=floatX))
y = test_value(y, np.ones((10, 15, 93), dtype=floatX))
x_mask = test_value(x_mask, np.ones((10, 15), dtype=floatX))
y_mask = test_value(y_mask, np.ones((10, 15), dtype=floatX))

# h = rnn_layer.apply(x)

inits = [IsotropicGaussian(0.001), IsotropicGaussian(0.001), IsotropicGaussian(0.001)]

def get_datastream(dataset, batch_size=160):
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

train = MidiSequence2('nottingham')
test = MidiSequence('nottingham', which_set='test')

train = get_datastream(train)
test = get_datastream(test, batch_size=256)

# srnn = SimpleRNN(dims=[93, 512, 93])
# srnn.initialize()
# hidden_states, cells, probs = srnn.apply(inputs=x, input_mask=x_mask)
# cost = NegativeLogLikelihood().apply(y, probs, y_mask)
#
# error_rate = MismulitclassificationRate().apply(y, probs, y_mask)
# error_rate.name = "error on note as a whole"
# mistake_rate = MismulitmistakeRate().apply(y, probs, y_mask)
# mistake_rate.name = "single error within note"

rnnrbm = Rnnrbm(93, 256, 93, 256)
rnnrbm.allocate()
rnnrbm.initialize()
cost, v_sample = rnnrbm.cost(x, x_mask)

error_rate = MismulitclassificationRate().apply(x, v_sample[-1], x_mask)
error_rate.name = "error on note as a whole"
mistake_rate = MismulitmistakeRate().apply(x, v_sample[-1], x_mask)
mistake_rate.name = "single error within note"

model = Model(cost)
cg = ComputationGraph([cost])

step_rule = CompositeRule([RemoveNotFinite(), StepClipping(20.0), Adam(learning_rate=.001), StepClipping(3.0),
                           RemoveNotFinite()])  # Scale(0.01)

gradients = dict(equizip(cg.parameters, T.grad(cost, cg.parameters, consider_constant=[v_sample])))
algorithm = GradientDescent(step_rule=step_rule, gradients=gradients, cost=cost, params=cg.parameters)
#
# algorithm = GradientDescent(step_rule=step_rule, cost=cost, params=cg.parameters)



## l2/l1 regularization
# reg = 0.000005
# params = VariableFilter(roles=[WEIGHT, BIAS])(cg.variables)
# param_nans = 0
# for i, p in enumerate(params):
# # cost += reg * abs(p).sum()
# cost += reg * (p ** 2).sum()
# param_nans += T.isnan(p).sum()
# name = params[i].name
#     params[i] = params[i].mean()
#     params[i].name = name + str(i)
cost.name = "final_cost"

# param_nans.name = 'params_nans'

## Training algorithm



# algorithm.initialize()
algorithm.add_updates(cg.updates)

extensions = [FinishAfter(after_n_epochs=5000),
              DataStreamMonitoring(
                  [cost, error_rate, mistake_rate],
                  data_stream=test,
                  updates=cg.updates,
                  prefix="test"),
              TrainingDataMonitoring(
                  [cost,  # hidden_states, debug_val, param_nans,
                   aggregation.mean(algorithm.total_gradient_norm)],  #+ params,
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
            #['train_hidden_states_NANS', 'train_params_nans'],
            ['test_error on note as a whole', 'test_single error within note'],
            ['test_final_cost', 'train_final_cost'],
            ['train_total_gradient_norm'],
            #['train_' + param.name for param in params],
        ]))

main_loop = MainLoop(algorithm=algorithm,
                     data_stream=train,
                     model=model,
                     extensions=extensions
                     )
main_loop.run()

newdir = str(datetime.now())
os.mkdir(newdir)
for i,param in enumerate(main_loop.model.parameters):
    np.save(os.path.join(newdir,param.name+str(i)), param.get_value())