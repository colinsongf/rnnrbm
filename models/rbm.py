__author__ = 'mike'

import sys

from blocks.bricks.recurrent import BaseRecurrent, recurrent
from blocks.bricks import Sigmoid, lazy, Initializable
from blocks.utils import shared_floatx_nans
from blocks.roles import add_role, WEIGHT, BIAS, has_roles
import theano
import theano.tensor as T
import numpy as np
from blocks.graph import ComputationGraph


sys.setrecursionlimit(10000)

from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

rng = RandomStreams(seed=np.random.randint(1 << 30))

floatX = theano.config.floatX


class Rbm(Initializable, BaseRecurrent):
    @lazy(allocation=['visible_dim', 'hidden_dim'])
    def __init__(self, visible_dim, hidden_dim, activation=Sigmoid(), **kwargs):
        super(Rbm, self).__init__(**kwargs)
        self.hidden_dim = hidden_dim
        self.visible_dim = visible_dim
        self.activation = activation
        self.children = [activation]

    @property
    def W(self):
        return self.params[0]

    @W.setter
    def W(self, value):
        self.params[0] = value

    @property
    def bv(self):
        return self.params[1]

    @property
    def bh(self):
        return self.params[2]

    def _allocate(self):
        W = shared_floatx_nans((self.visible_dim, self.hidden_dim), name='Wrbm')
        add_role(W, WEIGHT)
        self.params.append(W)
        self.add_auxiliary_variable(W.norm(2), name='W_norm')
        bv = shared_floatx_nans((self.visible_dim,), name='bv')
        add_role(bv, BIAS)
        self.params.append(bv)
        self.add_auxiliary_variable(bv.norm(2), name='bv_norm')
        bh = shared_floatx_nans((self.hidden_dim,), name='bh')
        add_role(bh, BIAS)
        self.params.append(bh)
        self.add_auxiliary_variable(bh.norm(2), name='bh_norm')


    def _initialize(self):
        for param in self.params:
            if has_roles(param, [WEIGHT]):
                self.weights_init.initialize(param, self.rng)
            elif has_roles(param, [BIAS]):
                self.biases_init.initialize(param, self.rng)

    @recurrent(sequences=[], states=['visible'], outputs=['mean_visible', 'visible'],
               contexts=['bv', 'bh', 'W'])
    def apply(self, visible=None, bv=None, bh=None):
        bv = self.bv if bv is None else bv
        bh = self.bh if bh is None else bh
        mean_hidden = self.activation.apply(T.dot(visible, self.W) + bh)
        h = rng.binomial(size=mean_hidden.shape, n=1, p=mean_hidden,
                         dtype=floatX)
        mean_visible = self.activation.apply(T.dot(h, self.W.T) + bv)
        v = rng.binomial(size=mean_visible.shape, n=1, p=mean_visible,
                         dtype=floatX)
        cg = ComputationGraph(v)
        return [mean_visible, v], cg.updates

    def free_energy(self, v, bv, bh, mask=None):
        bv = self.bv if bv is None else bv
        bh = self.bh if bh is None else bh
        if mask is not None:
            return -(v * bv * mask[:, :, None]).sum() - T.sum(
                mask[:, :, None] * T.log(1 + (T.exp(T.dot(v, self.W) + bh))))
        else:
            return -(v * bv).sum() - T.sum(T.log(1 + (T.exp(T.dot(v, self.W) + bh))))

    def cost(self, visible, k, batch_size, bv=None, bh=None, mask=None):
        _, v_sample = self.apply(visible=visible, bv=bv, bh=bh, n_steps=k, batch_size=batch_size)
        cost = (self.free_energy(visible, bv, bh, mask=mask) - self.free_energy(v_sample[-1], bv, bh,
                                                                                mask=mask)) / \
               visible.shape[0]
        return cost.astype(floatX), v_sample

    def get_dim(self, name):
        if name == 'visible':
            return self.visible_dim
        return super(Rbm, self).get_dim(name)
