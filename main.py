__author__ = 'mike'

import sys
import os
from datetime import datetime
from numpy import float32

from blocks.extensions.training import SharedVariableModifier
from blocks.algorithms import GradientDescent, Adam, CompositeRule, StepClipping, RemoveNotFinite
from blocks.model import Model
from blocks.extensions import Timing, Printing, FinishAfter, ProgressBar
from blocks.extensions.monitoring import TrainingDataMonitoring, DataStreamMonitoring
from blocks.extensions.plot import Plot
from picklable_itertools.extras import equizip
import theano
from theano import config
from blocks.graph import ComputationGraph
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
import numpy as np
from fuel.streams import DataStream
from fuel.schemes import ShuffledScheme
from fuel.transformers import Padding, Mapping
from theano import tensor as T
from blocks.main_loop import MainLoop

from midi import MidiSequence
from rbm_pretrain import get_rbm_pretraining_params, get_rnnrbm_training_params
from utils import test_value
from midi import MidiSequence2
from miditools.utils import midiwrite


floatX = config.floatX
rng = RandomStreams(seed=np.random.randint(1 << 30))
sys.setrecursionlimit(10000)
np.set_printoptions(threshold='nan')


def pretrain(train, x, x_mask, epochs=900, test=None):
    cdk = theano.shared(1)
    lr = theano.shared(float32(0.1))
    rbm, cost, v_sample, error_rate, mistake_rate = get_rbm_pretraining_params(x, x_mask, cdk=cdk)
    model = Model(cost)
    cg = ComputationGraph([cost])
    step_rule = CompositeRule([RemoveNotFinite(), StepClipping(30.0), Adam(learning_rate=lr), StepClipping(6.0),
                               RemoveNotFinite()])

    gradients = dict(equizip(cg.parameters, T.grad(cost, cg.parameters, consider_constant=[v_sample])))
    algorithm = GradientDescent(step_rule=step_rule, gradients=gradients, cost=cost, params=cg.parameters)
    algorithm.add_updates(cg.updates)
    extensions = [SharedVariableModifier(parameter=cdk, function=lambda n, v: 2 * v + 1 if n % (200 * 5) == 0 else v),
                  SharedVariableModifier(parameter=lr, function=lambda n, v: float32(0.7 * v) if n % 700 == 0 else v),
                  FinishAfter(after_n_epochs=epochs),
                  TrainingDataMonitoring(
                      [cost, error_rate, mistake_rate, ],  # hidden_states, debug_val, param_nans,
                      # aggregation.mean(algorithm.total_gradient_norm)],  #+ params,
                      prefix="train",
                      after_epoch=False, every_n_batches=10),
                  Timing(),
                  Printing(),
                  ProgressBar()]
    if test is not None:
        extensions.append(DataStreamMonitoring(
            [cost, error_rate, mistake_rate],
            data_stream=test,
            updates=cg.updates,
            prefix="test", every_n_batches=10))

    bokeh = True
    if bokeh:
        extensions.append(Plot(
            'Pretrain RBM',
            channels=[
                ['train_error on note as a whole', 'train_single error within note', 'test_error on note as a whole',
                 'test_single error within note'],
                ['train_rbm_cost'],
                # ['train_total_gradient_norm'],
            ]))

    main_loop = MainLoop(algorithm=algorithm,
                         data_stream=train,
                         model=model,
                         extensions=extensions
                         )
    main_loop.run()
    return main_loop, rbm


def train_rnnrbm(train, x, x_mask, epochs=1000, rbm=None, test=None):
    rnnrbm, cost, v_sample, error_rate, mistake_rate = get_rnnrbm_training_params(x, x_mask, epochs=epochs, rbm=rbm,
                                                                                  test_stream=test)
    cdk = theano.shared(10)
    lr = theano.shared(float32(0.004))

    model = Model(cost)
    cg = ComputationGraph([cost])
    step_rule = CompositeRule([RemoveNotFinite(), StepClipping(30.0), Adam(learning_rate=lr), StepClipping(6.0),
                               RemoveNotFinite()])  # Scale(0.01)
    gradients = dict(equizip(cg.parameters, T.grad(cost, cg.parameters, consider_constant=[v_sample])))
    algorithm = GradientDescent(step_rule=step_rule, gradients=gradients, cost=cost, params=cg.parameters)
    algorithm.add_updates(cg.updates)
    extensions = [
        SharedVariableModifier(parameter=cdk, function=lambda n, v: int(1.45 * v) if n % (200 * 5) == 0 else v),
        SharedVariableModifier(parameter=lr, function=lambda n, v: float32(0.78 * v) if n % (200 * 5) == 0 else v),
        FinishAfter(after_n_epochs=epochs),
        TrainingDataMonitoring(
            [cost, error_rate, mistake_rate, ],  # hidden_states, debug_val, param_nans,
            # aggregation.mean(algorithm.total_gradient_norm)],  #+ params,
            prefix="train",
            after_epoch=False, every_n_batches=10),
        Timing(),
        Printing(),
        ProgressBar()]
    if test is not None:
        extensions.append(DataStreamMonitoring(
            [cost, error_rate, mistake_rate],
            data_stream=test,
            updates=cg.updates,
            prefix="test", every_n_batches=10))
    bokeh = True
    if bokeh:
        extensions.append(Plot(
            'Training RNN-RBM',
            channels=[
                ['train_error on note as a whole', 'train_single error within note', 'test_error on note as a whole',
                 'test_single error within note'],
                ['train_final_cost'],
                # ['train_total_gradient_norm'],
            ]))

    main_loop = MainLoop(algorithm=algorithm,
                         data_stream=train,
                         model=model,
                         extensions=extensions
                         )
    main_loop.run()
    return main_loop, rnnrbm


def get_data(train_batch=160, test_batch=256):
    x = T.tensor3('features')
    x_mask = T.matrix('features_mask')
    y = T.tensor3('targets')
    y_mask = T.matrix('targets_mask')

    x = test_value(x, np.ones((10, 15, 93), dtype=floatX))
    y = test_value(y, np.ones((10, 15, 93), dtype=floatX))
    x_mask = test_value(x_mask, np.ones((10, 15), dtype=floatX))
    y_mask = test_value(y_mask, np.ones((10, 15), dtype=floatX))

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

    train_datastream = MidiSequence2('nottingham')
    test_datastream = MidiSequence('nottingham', which_set='test')

    train = get_datastream(train_datastream, batch_size=train_batch)
    test = get_datastream(test_datastream, batch_size=test_batch)

    return x, x_mask, y, y_mask, train, test


if __name__ == "__main__":
    x, x_mask, y, y_mask, train, test = get_data(train_batch=256, test_batch=256)
    pretrain_main, rbm = pretrain(train, x, x_mask, epochs=1200, test=test)

    newdir = str(datetime.now())
    os.mkdir(newdir)
    for i, param in enumerate(pretrain_main.model.parameters):
        np.save(os.path.join(newdir, param.name + str(i)), param.get_value())

    x, x_mask, y, y_mask, train, test = get_data(train_batch=160, test_batch=256)
    training_main, rnnrbm = train_rnnrbm(train, x, x_mask, epochs=1000, rbm=rbm, test=test)

    for i, param in enumerate(training_main.model.parameters):
        np.save(os.path.join(newdir, param.name + str(i)), param.get_value())

    generated_songs = rnnrbm.generate(n_steps=1000, batch_size=3, rbm_steps=100)[0]
    generate = theano.function([], [generated_songs], updates=ComputationGraph(generated_songs).updates)
    piano_rolls = generate()[0]
    print piano_rolls.shape
    piano_rolls = np.rollaxis(piano_rolls, 1, 0)

    for i, piano_roll in enumerate(piano_rolls):
        np.save(os.path.join(newdir, 'piano_roll%s' % i), piano_roll)
        midiwrite(newdir + '%s.midi' % i, piano_roll, r=(0, 93))
