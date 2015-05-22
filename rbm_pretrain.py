__author__ = 'mike'

import sys

from blocks.bricks import Sigmoid
from blocks.initialization import Constant
import itertools
import theano
import numpy as np
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
from blocks.initialization import IsotropicGaussian

from models import Rbm, Rnnrbm
from utils import MismulitclassificationRate, MismulitmistakeRate

rng = RandomStreams(seed=np.random.randint(1 << 30))
floatX = theano.config.floatX
sys.setrecursionlimit(10000)


def initialize_rbm(Wrbm=None, bh=None, bv=None):
    rbm = Rbm(visible_dim=93, hidden_dim=256,
              activation=Sigmoid(), weights_init=IsotropicGaussian(0.01), biases_init=Constant(0.1),
              name='rbm2l')
    rbm.allocate()
    rbm.initialize()

    if Wrbm is not None:
        rbm.W.set_value(Wrbm)
    if bv is not None:
        rbm.bv.set_value(bv)
    if bh is not None:
        rbm.bh.set_value(bh)

    return rbm

def initialize_rnnrbm(rbm=None, **kwargs):
    rnnrbm = Rnnrbm(256, 93, 256)
    rnnrbm.allocate()
    rnnrbm.initialize(pretrained_rbm=rbm)
    # params = list(itertools.chain(*[child.params for child in rnnrbm.children]))
    for child in rnnrbm.children:
        for param in child.params:
            print param.name , " -> ",
            if param.name in kwargs:
                print "in kwargs"
                param.set_value(kwargs[param.name])
            else:
                print "not in kwargs"

    rnnrbm.rbm.bh.set_value(kwargs['buh'])
    rnnrbm.rbm.bv.set_value(kwargs['buv'])

    return rnnrbm


# noinspection PyTypeChecker,PyTypeChecker
def get_rbm_pretraining_params(x, x_mask, cdk=1):
    rbm = initialize_rbm()
    cost, v_sample = rbm.cost(visible=x, k=cdk, batch_size=x.shape[0], mask=x_mask)
    error_rate = MismulitclassificationRate().apply(x, v_sample[-1], x_mask)
    error_rate.name = "error on note as a whole"
    mistake_rate = MismulitmistakeRate().apply(x, v_sample[-1], x_mask)
    mistake_rate.name = "single error within note"

    cost.name = 'rbm_cost'
    return rbm, cost, v_sample, error_rate, mistake_rate


def get_rnnrbm_training_params(x, x_mask, rbm=None, cdk=10, rnnrbm=None):
    if rnnrbm is None:
        rnnrbm = initialize_rnnrbm(rbm=rbm)
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