__author__ = 'mike'

import theano
from theano import ifelse
from theano import tensor as T
from blocks.bricks.cost import Cost
from blocks.bricks.base import application, Brick

floatX = theano.config.floatX


class MismulitclassificationRate(Cost):
    @application(outputs=["error_rate"])
    def apply(self, y, y_hat, y_mask=None):
        if y_mask:
            result = (T.sum(T.max(T.neq(y, T.ge(y_hat, 0.5)), axis=2) * y_mask) /
                      (T.sum(y_mask).astype(floatX)))
        else:
            result = (T.sum(T.max(T.neq(y, T.ge(y_hat, 0.5)), axis=2)) /
                      (y.shape[0] * y.shape[1]).astype(floatX))

        return result


class NegativeLogLikelihood(Cost):
    @application(outputs=["cost"])
    def apply(self, y, y_hat, y_mask):
        return T.mean(-T.sum(y * T.log(abs(y_hat)+1e-14), axis=2)*y_mask)