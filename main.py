import sys
import os
import argparse
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
from rbm_pretrain import get_rbm_pretraining_params, get_rnnrbm_training_params, initialize_rbm, initialize_rnnrbm
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


def train_rnnrbm(train, x, x_mask, epochs=1000, rbm=None, rnnrbm=None, test=None):
    cdk = theano.shared(10)
    lr = theano.shared(float32(0.004))

    rnnrbm, cost, v_sample, error_rate, mistake_rate = get_rnnrbm_training_params(x, x_mask, rbm=rbm, cdk=cdk, rnnrbm=rnnrbm)

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

    parser = argparse.ArgumentParser(description="Train RnnRbm")
    parser.add_argument('--rbm', type=str, help='Rbm params')
    parser.add_argument('--rnnrbm', type=str, help='Rnnrbm params')
    parser.add_argument('--train', type=str, help='Rnnrbm params')
    parser.add_argument('--save', type=str, help='Rnnrbm params')


    args = parser.parse_args()
    rbm = None
    rnnrbm = None

    newdir = datetime.now().isoformat().replace(':', '-')
    if args.save:
        os.mkdir(newdir)

    if args.rnnrbm:
        params = {

            "Wrbm": np.load(os.path.join(args.rnnrbm, 'Wrbm.npy')),
            "Wuv": np.load(os.path.join(args.rnnrbm, 'Wuv.npy')),
            "Wuh": np.load(os.path.join(args.rnnrbm, 'Wuh.npy')),
            "Wvu": np.load(os.path.join(args.rnnrbm, 'Wvu.npy')),
            "buv": np.load(os.path.join(args.rnnrbm, 'buv.npy')),
            "buh": np.load(os.path.join(args.rnnrbm, 'buh.npy')),
            "bvu": np.load(os.path.join(args.rnnrbm, 'bvu.npy')),
            "W_cell_to_forget": np.load(os.path.join(args.rnnrbm, 'W_cell_to_forget.npy')),
            "W_cell_to_in": np.load(os.path.join(args.rnnrbm, 'W_cell_to_in.npy')),
            "W_cell_to_out": np.load(os.path.join(args.rnnrbm, 'W_cell_to_out.npy')),
            "W_state": np.load(os.path.join(args.rnnrbm, 'W_state.npy'))
        }
        rnnrbm = initialize_rnnrbm(**params)

    elif args.rbm:
        print "Retrieving pretrained RBM"
        Wrbm = np.load(os.path.join(args.rbm, 'Wrbm.npy'))
        bv = np.load(os.path.join(args.rbm, 'bv.npy'))
        bh = np.load(os.path.join(args.rbm, 'bh.npy'))
        rbm = initialize_rbm(Wrbm=Wrbm, bv=bv, bh=bh)
    else:
        print "Pretraining Rbm"
        x, x_mask, y, y_mask, train, test = get_data(train_batch=256, test_batch=256)
        pretrain_main, rbm = pretrain(train, x, x_mask, epochs=1200, test=test)
        if args.save:
            for i, param in enumerate(pretrain_main.model.parameters):
                np.save(os.path.join(newdir, param.name), param.get_value())

    x, x_mask, y, y_mask, train, test = get_data(train_batch=160, test_batch=256)

    if args.train:
        training_main, rnnrbm = train_rnnrbm(train, x, x_mask, epochs=1000, rbm=rbm, rnnrbm=rnnrbm, test=test)
        if args.save:
            for i, param in enumerate(training_main.model.parameters):
                np.save(os.path.join(newdir, param.name), param.get_value())

    examples = []
    for t in train.get_epoch_iterator():
        examples.append(t[0][:40, 3:5, :])

    examples = np.hstack(examples)[:40]

    burn_in_input = T.tensor3('burn_in')
    hidden_state, cells, _, _ = rnnrbm.training_biases(visible=burn_in_input)

    generated_songs = rnnrbm.generate(visible=burn_in_input[-1], hidden_state=hidden_state[-1], cells=cells[-1],
                                      n_steps=1000, batch_size=examples.shape[1],
                                      rbm_steps=30)[0]
    generated_songs = T.concatenate((burn_in_input, generated_songs))
    # burn_in = theano.function([x],[hidden_state,cells])
    generate = theano.function([burn_in_input], [generated_songs], updates=ComputationGraph(generated_songs).updates)
    piano_rolls = generate(examples)[0]
    print piano_rolls.shape
    piano_rolls = np.rollaxis(piano_rolls, 1, 0)

    for i, piano_roll in enumerate(piano_rolls):
        if args.save:
             np.save(os.path.join(newdir, 'piano_roll%s' % i), piano_roll)
        midiwrite(newdir + '%s.midi' % i, piano_roll, r=(0, 93))
