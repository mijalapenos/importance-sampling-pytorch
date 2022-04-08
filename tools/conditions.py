# https://github.com/idiap/importance-sampling/blob/master/importance_sampling/samplers.py
import numpy as np


class Condition(object):
    """An interface for use with the ConditionalStartSampler."""

    @property
    def satisfied(self):
        raise NotImplementedError()

    @property
    def previously_satisfied(self):
        pass  # not necessary

    def update(self, scores):
        pass  # not necessary


class WarmupCondition(Condition):
    """Wait 'warmup' iterations before using importance sampling.

    Arguments
    ---------
        warmup: int
                The number of iterations to wait before starting importance
                sampling
    """

    def __init__(self, warmup=100):
        self._warmup = warmup
        self._iters = 0

    @property
    def satisfied(self):
        return self._iters > self._warmup

    def update(self, scores):
        self._iters += 1


class ExpCondition(Condition):
    """Assume that the scores are created by an exponential distribution and
    sample only if lamda is larger than x.

    Arguments
    ---------
        lambda_th: float
                   When lambda > lambda_th start importance sampling
        momentum: float
                  The momentum to compute the exponential moving average of
                  lambda
    """

    def __init__(self, lambda_th=2.0, momentum=0.9):
        self._lambda_th = lambda_th
        self._lambda = 0.0
        self._previous_lambda = 0.0
        self._momentum = momentum

    @property
    def satisfied(self):
        self._previous_lambda = self._lambda
        return self._lambda > self._lambda_th

    @property
    def previously_satisfied(self):
        return self._previous_lambda > self._lambda_th

    def update(self, scores):
        self._lambda = (
                self._momentum * self._lambda +
                (1 - self._momentum) / scores.mean()
        )


class TotalVariationCondition(Condition):
    """Sample from the decorated sampler if the TV of the scores with the
    uniform distribution is larger than a given value.

    Arguments
    ---------
        tv_th: float
               When tv > tv_th start importance sampling
        momentum: float
                  The momentum to compute the exponential moving average of
                  tv
    """

    def __init__(self, tv_th=0.5, momentum=0.9):
        self._tv_th = tv_th
        self._tv = 0.0
        self._previous_tv = 0.0
        self._momentum = momentum

    @property
    def satisfied(self):
        self._previous_tv = self._tv
        return self._tv > self._tv_th

    @property
    def previously_satisfied(self):
        return self._previous_tv > self._tv_th

    def update(self, scores):
        self._previous_tv = self._tv
        new_tv = 0.5 * np.abs(scores / scores.sum() - 1.0 / len(scores)).sum()
        self._tv = (
                self._momentum * self._tv +
                (1 - self._momentum) * new_tv
        )


class VarianceReductionCondition(Condition):
    """Sample with importance sampling when the variance reduction is larger
    than a threshold. The variance reduction units are in batch size increment.

    Arguments
    ---------
        vr_th: float
               When vr > vr_th start importance sampling
        momentum: float
                  The momentum to compute the exponential moving average of
                  vr
    """
    def __init__(self, vr_th=1.2, momentum=0.9):
        self._vr_th = vr_th
        self._vr = 0.0
        self._previous_vr = 0.0
        self._momentum = momentum

    @property
    def variance_reduction(self):
        return self._vr

    @property
    def satisfied(self):
        self._previous_vr = self._vr
        return self._vr > self._vr_th

    @property
    def previously_satisfied(self):
        return self._previous_vr > self._vr_th

    def update(self, scores):
        u = 1.0/len(scores)
        S = scores.sum().cpu()
        if S == 0:
            g = np.array(u)
        else:
            g = scores/S
        new_vr = 1.0 / np.sqrt(1 - ((g-u)**2).sum()/(g**2).sum())
        self._vr = (
            self._momentum * self._vr +
            (1-self._momentum) * new_vr
        )



class RewrittenCondition(Condition):
    """Sample with importance sampling when the variance reduction is larger
    than a threshold. The variance reduction units are in batch size increment.

    Arguments
    ---------
        vr_th: float
               When vr > vr_th start importance sampling
        momentum: float
                  The momentum to compute the exponential moving average of
                  vr
    """
    def __init__(self, vr_th=1.2, momentum=0.9):
        self._vr_th = vr_th
        self._vr = 0.0
        self._previous_vr = 0.0
        self._momentum = momentum

    @property
    def variance_reduction(self):
        return self._vr

    @property
    def satisfied(self):
        self._previous_vr = self._vr
        return self._vr > self._vr_th

    @property
    def previously_satisfied(self):
        return self._previous_vr > self._vr_th

    def update(self, scores):
        u = 1.0 / len(scores)
        S = scores.sum().cpu()
        if S == 0:
            g = np.array(u)  # no scores means uniform
        else:
            g = scores/S  # now we have probs
        new_vr = 1.0 / np.sqrt(1 - ((g-u)**2).sum()/(g**2).sum())
        self._vr = (
            self._momentum * self._vr +
            (1-self._momentum) * new_vr
        )
