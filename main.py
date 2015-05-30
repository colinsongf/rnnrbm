import sys
import os
import argparse
from datetime import datetime
from numpy import float32
from collections import OrderedDict

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
from blocks.dump import load_parameter_values, save_parameter_values

from utils import MismulitclassificationRate, MismulitmistakeRate, NegativeLogLikelihood
from models.rnnrbm import Rnnrbm
from utils import test_value
from midi import MidiSequence2, MidiSequence
from miditools.utils import midiwrite


floatX = config.floatX
rng = RandomStreams(seed=np.random.randint(1 << 30))
sys.setrecursionlimit(10000)
np.set_printoptions(threshold='nan')

x = T.tensor3('features')
x_mask = T.matrix('features_mask')
y = T.tensor3('targets')
y_mask = T.matrix('targets_mask')

x = test_value(x, np.ones((10, 15, 88), dtype=floatX))
y = test_value(y, np.ones((10, 15, 88), dtype=floatX))
x_mask = test_value(x_mask, np.ones((10, 15), dtype=floatX))
y_mask = test_value(y_mask, np.ones((10, 15), dtype=floatX))

pre_training_params = {
    '/rbm.buh': '/rnnrbm/uh.buh',
    '/rbm.Wrbm': '/rnnrbm/rbm.Wrbm',
    '/rbm.buv': '/rnnrbm/uv.buv',
    # '/rnn/out_layer/linear_0.b',
    # '/rnn/out_layer/linear_0.W',
    # '/rnnrbm/uh.Wuh',
    # '/rnnrbm/uv.Wuv',
    '/rnn/lstm_rnn_layer.W_cell_to_out': '/rnnrbm/lstm_rnn_layer.W_cell_to_out',
    '/rnn/lstm_rnn_layer.W_cell_to_in': '/rnnrbm/lstm_rnn_layer.W_cell_to_in',
    '/rnn/lstm_rnn_layer.W_cell_to_forget': '/rnnrbm/lstm_rnn_layer.W_cell_to_forget',
    '/rnn/h2h_transform.W': '/rnnrbm/h2h_transform.W',
    '/rnn/gru_rnn_layer.W': '/rnnrbm/gru_rnn_layer.W',
    '/rnn/input_transfrom.W': '/rnnrbm/input_transfrom.W',
    '/rnn/lstm_rnn_layer.W_state': '/rnnrbm/lstm_rnn_layer.W_state',
}

rbm_cdk = {
    3000: 3,
    6000: 7,
    9000: 12,
    9500: 20,
    9700: 25,
    9800: 30,
    9900: 40,
    9950: 50,
    9980: 70,
    9990: 80
}

rnnrbm_cdk = {
    5000: 20,
    10000: 30,
    10500: 40,
    11000: 50,
    11500: 60,
    12000: 80,
    12500: 90

}




def pretrain_rnn(train, rnnrbm, test=None, epochs=1000, bokeh=True):
    lr = theano.shared(float32(0.1))

    probs, _, _, _ = rnnrbm.rnn_pretrain_pred(x, x_mask)
    cost = NegativeLogLikelihood().apply(y, probs, y_mask)

    error_rate = MismulitclassificationRate().apply(y, probs, y_mask)
    error_rate.name = "error on note as a whole"
    mistake_rate = MismulitmistakeRate().apply(y, probs, y_mask)
    mistake_rate.name = "single error within note"
    cost.name = 'final_cost'

    model = Model(cost)
    cg = ComputationGraph([cost])
    step_rule = CompositeRule(
        [RemoveNotFinite(), StepClipping(30.0), Adam(learning_rate=lr), StepClipping(6.0),
         RemoveNotFinite()])
    algorithm = GradientDescent(step_rule=step_rule, cost=cost, params=cg.parameters)
    extensions = [SharedVariableModifier(parameter=lr,
                                         function=lambda n, v: float32(0.7 * v) if n % 700 == 0 else v),
                  FinishAfter(after_n_epochs=epochs),
                  TrainingDataMonitoring(
                      [cost, error_rate, mistake_rate, ],  # hidden_states, debug_val, param_nans,
                      # aggregation.mean(algorithm.total_gradient_norm)],  #+ params,
                      prefix="train",
                      after_epoch=False, every_n_batches=40),
                  Timing(),
                  Printing(),
                  ProgressBar()]
    if test is not None:
        extensions.append(DataStreamMonitoring(
            [cost, error_rate, mistake_rate],
            data_stream=test,
            updates=cg.updates,
            prefix="test", after_epoch=False, every_n_batches=40))

    if bokeh:
        extensions.append(Plot(
            'Pretrain RNN',
            channels=[
                ['train_error on note as a whole', 'train_single error within note',
                 'test_error on note as a whole',
                 'test_single error within note'],
                ['train_rbm_cost'],
                # ['train_total_gradient_norm'],
            ]))

    main_loop = MainLoop(algorithm=algorithm,
                         data_stream=train,
                         model=model,
                         extensions=extensions
                         )
    return main_loop


def pretrain_rbm(train, rnnrbm, test=None, epochs=900, bokeh=True, load_path=None):
    cdk = theano.shared(1)
    lr = theano.shared(float32(0.1))

    cost, v_sample = rnnrbm.rbm.cost(visible=x, k=cdk, batch_size=x.shape[0], mask=x_mask)
    error_rate = MismulitclassificationRate().apply(x, v_sample[-1], x_mask)
    error_rate.name = "error on note as a whole"
    mistake_rate = MismulitmistakeRate().apply(x, v_sample[-1], x_mask)
    mistake_rate.name = "single error within note"

    cost.name = 'rbm_cost'
    model = Model(cost)
    cg = ComputationGraph([cost])
    step_rule = CompositeRule(
        [RemoveNotFinite(), StepClipping(30.0), Adam(learning_rate=lr), StepClipping(6.0),
         RemoveNotFinite()])
    gradients = dict(equizip(cg.parameters, T.grad(cost, cg.parameters, consider_constant=[v_sample])))
    algorithm = GradientDescent(step_rule=step_rule, gradients=gradients, cost=cost,
                                params=cg.parameters)
    algorithm.add_updates(cg.updates)
    extensions = [SharedVariableModifier(parameter=cdk,
                                         function=lambda n, v: rbm_cdk[n] if rbm_cdk.get(n) else v),
                  SharedVariableModifier(parameter=lr,
                                         function=lambda n, v: float32(0.7 * v) if n % 1500 == 0 else v),
                  FinishAfter(after_n_epochs=epochs),
                  TrainingDataMonitoring(
                      [cost, error_rate, mistake_rate, ],  # hidden_states, debug_val, param_nans,
                      # aggregation.mean(algorithm.total_gradient_norm)],  #+ params,
                      prefix="train",
                      after_epoch=False, every_n_batches=40),
                  Timing(),
                  Printing(),
                  ProgressBar()]
    if test is not None:
        extensions.append(DataStreamMonitoring(
            [cost, error_rate, mistake_rate],
            data_stream=test,
            updates=cg.updates,
            prefix="test", after_epoch=False, every_n_batches=40))

    if bokeh:
        extensions.append(Plot(
            'Pretrain RBM',
            channels=[
                ['train_error on note as a whole', 'train_single error within note',
                 'test_error on note as a whole',
                 'test_single error within note'],
                ['train_rbm_cost'],
                # ['train_total_gradient_norm'],
            ]))

    main_loop = MainLoop(algorithm=algorithm,
                         data_stream=train,
                         model=model,
                         extensions=extensions
                         )
    return main_loop


def train_rnnrbm(train, rnnrbm, epochs=1000, test=None, bokeh=True,
                 load_path=None):
    cdk = theano.shared(10)
    lr = theano.shared(float32(0.004))

    cost, v_sample = rnnrbm.cost(examples=x, mask=x_mask, k=cdk)

    error_rate = MismulitclassificationRate().apply(x, v_sample[-1], x_mask)
    error_rate.name = "error on note as a whole"
    mistake_rate = MismulitmistakeRate().apply(x, v_sample[-1], x_mask)
    mistake_rate.name = "single error within note"
    cost.name = 'rbm_cost'

    model = Model(cost)
    cg = ComputationGraph([cost])
    step_rule = CompositeRule(
        [RemoveNotFinite(), StepClipping(30.0), Adam(learning_rate=lr), StepClipping(6.0),
         RemoveNotFinite()])  # Scale(0.01)
    gradients = dict(equizip(cg.parameters, T.grad(cost, cg.parameters, consider_constant=[v_sample])))
    algorithm = GradientDescent(step_rule=step_rule, gradients=gradients, cost=cost,
                                params=cg.parameters)
    algorithm.add_updates(cg.updates)
    extensions = [
        SharedVariableModifier(parameter=cdk,
                               function=lambda n, v: rnnrbm_cdk[n] if rnnrbm_cdk.get(n) else v),
        SharedVariableModifier(parameter=lr,
                               function=lambda n, v: float32(0.78 * v) if n % (200 * 5) == 0 else v),
        FinishAfter(after_n_epochs=epochs),
        TrainingDataMonitoring(
            [cost, error_rate, mistake_rate, ],  # hidden_states, debug_val, param_nans,
            # aggregation.mean(algorithm.total_gradient_norm)],  #+ params,
            prefix="train",
            after_epoch=False, every_n_batches=40),
        Timing(),
        Printing(),
        ProgressBar()]
    if test is not None:
        extensions.append(DataStreamMonitoring(
            [cost, error_rate, mistake_rate],
            data_stream=test,
            updates=cg.updates,
            prefix="test", after_epoch=False, every_n_batches=40))
    if bokeh:
        extensions.append(Plot(
            'Training RNN-RBM',
            channels=[
                ['train_error on note as a whole', 'train_single error within note',
                 'test_error on note as a whole',
                 'test_single error within note'],
                ['train_final_cost'],
                # ['train_total_gradient_norm'],
            ]))

    main_loop = MainLoop(algorithm=algorithm,
                         data_stream=train,
                         model=model,
                         extensions=extensions
                         )
    return main_loop


def get_data(train_batch=160, test_batch=256, datasets=('jsb',), rbm=True):
    def get_datastream(dataset, batch_size=160):
        dataset = DataStream(
            dataset,
            iteration_scheme=ShuffledScheme(
                dataset.num_examples, batch_size
            ),
        )
        dataset = Padding(dataset)

        # if flatten:
        # dataset = Flatten(dataset, which_sources=('features,'))

        def _transpose(data):
            return tuple(np.rollaxis(array, 1, 0) for array in data)

        dataset = Mapping(dataset, _transpose)
        return dataset

    train_datastream = MidiSequence2(datasets) if rbm else MidiSequence(datasets)
    test_datastream = MidiSequence2(datasets, which_set='test') if rbm else MidiSequence(datasets)
    train = get_datastream(train_datastream, batch_size=train_batch)
    test = get_datastream(test_datastream, batch_size=test_batch)
    return train, test


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Train RnnRbm")
    parser.add_argument('--rbm', action='store_true', help='Rbm params')
    parser.add_argument('--rnn', action='store_true', help='Rbm params')
    parser.add_argument('--rnnrbm', action='store_true', help='Rnnrbm params')
    parser.add_argument('--train', action='store_true', help='Rnnrbm params')
    parser.add_argument('--bokeh', action='store_true', help='Rnnrbm params')
    parser.add_argument('--model', type=str, help='Rnnrbm params')
    parser.add_argument('--save', action='store_true', help='Rnnrbm params')


    rbm_epochs, rnn_epochs, rnnrbm_epochs = 1000, 600, 500
    args = parser.parse_args()

    rnnrbm = Rnnrbm(88, 256, (350, 250), name='rnnrbm')
    rnnrbm.allocate()
    rnnrbm.initialize()

    params = OrderedDict()
    if args.model:
        params = load_parameter_values(args.model)
    newdir = datetime.now().isoformat().replace(':', '-')

    def run_main(main_loop, params=None):
        if bool(params):
            print "setting up params"
            main_loop.model.set_param_values(params)
        main_loop.run()
        params.update(main_loop.model.get_param_values())
        for key, value in dict(params).iteritems():
            if key in pre_training_params:
                new_key = pre_training_params[key]
                params[new_key] = params[key]
        return params

    datasets = ('midi', 'nottingham', 'muse', 'jsb')

    if args.rbm:
        print "Training RBM"
        train, test = get_data(train_batch=160, test_batch=140, datasets=datasets)
        main_loop = pretrain_rbm(train, rnnrbm, test, epochs=rbm_epochs, bokeh=args.bokeh)
        params = run_main(main_loop, params)

    if args.rnn:
        print "Training RNN"
        train, test = get_data(train_batch=80, test_batch=120, rbm=False, datasets=datasets)
        main_loop = pretrain_rnn(train, rnnrbm, test, epochs=rnn_epochs, bokeh=args.bokeh)
        params = run_main(main_loop, params)

    if args.rnnrbm:
        print "Training RNN-RBM"
        train, test = get_data(train_batch=60, test_batch=120, datasets=datasets)
        main_loop = train_rnnrbm(train, rnnrbm, test=test, epochs=rnnrbm_epochs, bokeh=args.bokeh)
        params = run_main(main_loop, params)

    if args.save:
        os.mkdir(newdir)
        save_parameter_values(params, os.path.join(newdir, "model.npz"))

    examples = []
    train, test = get_data(train_batch=100, test_batch=100)
    for t in train.get_epoch_iterator():
        examples.append(t[0][:40, 3:5, :])

    examples = np.hstack(examples)[:40]
    song_size = T.iscalar('song_size')
    iters = T.iscalar('rbm_iters')
    burn_in_input = T.tensor3('burn_in')
    gru_state, lstm_state, lstm_cells, _, _ = rnnrbm.training_biases(visible=burn_in_input)

    generated_songs = \
        rnnrbm.generate(visible=burn_in_input[-1], gru_state=gru_state[-1], lstm_state=lstm_state[-1],
                        lstm_cells=lstm_cells[-1],
                        n_steps=song_size, batch_size=examples.shape[1],
                        rbm_steps=150)[0]
    generated_songs = T.concatenate((burn_in_input, generated_songs))
    generate = theano.function([burn_in_input, song_size], [generated_songs],
                               updates=ComputationGraph(generated_songs).updates)
    piano_rolls = generate(examples, 500)[0]
    print piano_rolls.shape
    piano_rolls = np.rollaxis(piano_rolls, 1, 0)

    for i, piano_roll in enumerate(piano_rolls):
        if args.save:
            np.save(os.path.join(newdir, 'piano_roll%s' % i), piano_roll)
        midiwrite(newdir + '%s.midi' % i, piano_roll)
