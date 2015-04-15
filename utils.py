__author__ = 'mike'

import theano
from theano.ifelse import ifelse
from theano import tensor as T
from blocks.bricks.cost import Cost
from blocks.bricks.base import application, Brick
from blocks.bricks import Activation
from blocks.roles import add_role, WEIGHT, BIAS

floatX = theano.config.floatX

threshold = 0.5

class MismulitclassificationRate(Cost):
    @application(outputs=["error_rate"])
    def apply(self, y, y_hat, y_mask=None):
        if y_mask:
            result = (T.sum(T.max(T.neq(y, T.ge(y_hat, threshold)), axis=2) * y_mask) /
                      (T.sum(y_mask).astype(floatX)))
        else:
            result = (T.sum(T.max(T.neq(y, T.ge(y_hat, threshold)), axis=2)) /
                      (y.shape[0] * y.shape[1]).astype(floatX))

        return result

class MismulitmistakeRate(Cost):
    @application(outputs=["error_rate"])
    def apply(self, y, y_hat, y_mask=None):
        if y_mask:
            result = (T.sum(T.sum(T.neq(y, T.ge(y_hat, threshold)), axis=2) * y_mask) /
                      ((T.sum(y_mask)*y.shape[2]).astype(floatX)))
        else:
            result = (T.sum(T.max(T.neq(y, T.ge(y_hat, threshold)), axis=2)) /
                      (y.shape[0] * y.shape[1]).astype(floatX))

        return result


class MeanSquare(Cost):
    @application(outputs=["cost"])
    def apply(self, y, y_hat, y_mask):
        return T.mean(T.sum((y - y_hat) ** 2, axis=2) * y_mask)

class MeanSquare(Cost):
    @application(outputs=["cost"])
    def apply(self, y, y_hat, y_mask):
        return T.mean(T.sum((y - y_hat) ** 2, axis=2) * y_mask)

class NanRectify(Activation):
    @application(inputs=['input_'], outputs=['output'])
    def apply(self, input_):
        input_ = T.switch(T.isnan(input_), 0, input_)
        return T.switch(input_ > 0, input_, 0.01*input_)

class ParametricRectifier(Activation):
    def __init__(self, leaky_init, *args, **kwargs):
        super(ParametricRectifier, self).__init__(*args, **kwargs)
        self.leaky_init = leaky_init
        add_role(self.leaky_init, WEIGHT)

    @application(inputs=['input_'], outputs=['output'])
    def apply(self, input_):
        res = T.switch(input_ > 0, input_, self.leaky_init * input_)
        return T.switch(T.isnan(res), 0, res)