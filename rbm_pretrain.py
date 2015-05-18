__author__ = 'mike'

import sys

from blocks.extensions.training import SharedVariableModifier
from theano import config
from blocks.bricks import Sigmoid
from blocks.initialization import Constant
import theano
import numpy as np
from models import Rbm, Rnnrbm
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
from theano import tensor as T
from blocks.initialization import IsotropicGaussian
from blocks.algorithms import GradientDescent, Adam, CompositeRule, StepClipping, RemoveNotFinite
from blocks.model import Model
from blocks.extensions import Timing, Printing, FinishAfter, ProgressBar
from blocks.extensions.monitoring import TrainingDataMonitoring, DataStreamMonitoring
from blocks.monitoring import aggregation
from blocks.extensions.plot import Plot
from picklable_itertools.extras import equizip

from blocks.graph import ComputationGraph
from utils import MismulitclassificationRate, MismulitmistakeRate

rng = RandomStreams(seed=np.random.randint(1 << 30))
floatX = theano.config.floatX
sys.setrecursionlimit(10000)

# noinspection PyTypeChecker,PyTypeChecker
def get_rbm_pretraining_params(x, x_mask, cdk=1):

    rbm = Rbm(visible_dim=93, hidden_dim=256,
              activation=Sigmoid(), weights_init=IsotropicGaussian(0.01), biases_init=Constant(0.1),
              name='rbm2l')
    rbm.allocate()
    rbm.initialize()

    cost, v_sample = rbm.cost(visible=x, k=cdk, batch_size=x.shape[0], mask=x_mask)

    error_rate = MismulitclassificationRate().apply(x, v_sample[-1], x_mask)
    error_rate.name = "error on note as a whole"
    mistake_rate = MismulitmistakeRate().apply(x, v_sample[-1], x_mask)
    mistake_rate.name = "single error within note"

    cost.name = 'rbm_cost'
    return rbm, cost, v_sample, error_rate, mistake_rate


def get_rnnrbm_training_params(x, x_mask, epochs=900, rbm=None, test_stream=None, cdk=10):
    rnnrbm = Rnnrbm(256, 93, 256)
    rnnrbm.allocate()
    rnnrbm.initialize(pretrained_rbm=rbm)
    cost, v_sample = rnnrbm.cost(examples=x, mask=x_mask, k=cdk)
    error_rate = MismulitclassificationRate().apply(x, v_sample[-1], x_mask)
    error_rate.name = "error on note as a whole"
    mistake_rate = MismulitmistakeRate().apply(x, v_sample[-1], x_mask)
    mistake_rate.name = "single error within note"


    ## l2/l1 regularization
    # reg = 0.000005
    # params = VariableFilter(roles=[WEIGHT, BIAS])(cg.variables)
    # param_nans = 0
    # for i, p in enumerate(params):
    # # cost += reg * abs(p).sum()
    # cost += reg * (p ** 2).sum()
    # param_nans += T.isnan(p).sum()
    # name = params[i].name
    # params[i] = params[i].mean()
    # params[i].name = name + str(i)
    cost.name = "final_cost"
    return rnnrbm, cost, v_sample, error_rate, mistake_rate