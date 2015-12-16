# ----------------------------------------------------------------------------
# Copyright 2014 Nervana Systems Inc.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ----------------------------------------------------------------------------
from neon import NervanaObject
import numpy as np


class Cost(NervanaObject):

    """
    Base class for the cost functions
    """

    def __call__(self, y, t):
        """
        Applies the cost function

        Args:
            y (Tensor or OpTree): Output of previous layer or model
            t (Tensor or OpTree): True targets corresponding to y

        Returns:
            OpTree: Returns the cost
        """
        return self.func(y, t)

    def bprop(self, y, t):
        """
        Computes the derivative of the cost function

        Args:
            y (Tensor or OpTree): Output of previous layer or model
            t (Tensor or OpTree): True targets corresponding to y

        Returns:
            OpTree: Returns the derivative of the cost function
        """
        return self.funcgrad(y, t)


class Metric(Cost):

    """
    Base class for Metric

    Meant for non-smooth costs that we just want to check on validation.
    """

    def __call__(self, y, t):
        """
        To implement in derived classes

        Args:
            y (Tensor or OpTree): Output of previous layer or model
            t (Tensor or OpTree): True targets corresponding to y

        Returns:
            float: Returns the metric
        """
        raise NotImplementedError()

    def bprop(self, y, t):
        """
        Not relevant for Metric
        """
        pass


class CrossEntropyBinary(Cost):

    """
    Applies the binary cross entropy function

    Note:
        bprop assumes that shortcut is used to calculate derivative
    """

    def __init__(self, scale=1):
        """
        Initialize the binary cross entropy function

        Args:
            scale (float): amount by which to scale the backpropagated error
        """
        self.scale = scale

    def __call__(self, y, t):
        """
        Applies the binary cross entropy cost function

        Args:
            y (Tensor or OpTree): Output of previous layer or model
            t (Tensor or OpTree): True targets corresponding to y

        Returns:
            OpTree: Returns the binary cross entropy cost
        """
        a = - self.be.safelog(y) * t
        b = - self.be.safelog(1 - y) * (1 - t)
        return self.be.sum(a + b, axis=0)

    def bprop(self, y, t):
        """
        Computes the shortcut derivative of the binary cross entropy cost function

        Args:
            y (Tensor or OpTree): Output of previous layer or model
            t (Tensor or OpTree): True targets corresponding to y

        Returns:
            OpTree: Returns the (mean) shortcut derivative of the binary entropy
                    cost function ``(y - t) / y.shape[1]``
        """
        return self.scale * (y - t)


class CrossEntropyMulti(Cost):

    """
    Applies the multiclass cross entropy function

    Note:
        bprop assumes that shortcut is used to calculate derivative
    """

    def __init__(self, scale=1, usebits=False):
        """
        Initialize the multiclass cross entropy function

        Args:
            scale (float): amount by which to scale the backpropagated error
            usebits (boolean): whether to display costs in bits or nats (default)
        """
        self.scale = scale
        self.logscale = np.float(1. / np.log(2.0) if usebits else 1.)

    def __call__(self, y, t):
        """
        Applies the multiclass cross entropy cost function

        Args:
            y (Tensor or OpTree): Output of previous layer or model
            t (Tensor or OpTree): True targets corresponding to y

        Returns:
            OpTree: Returns the multiclass cross entropy cost
        """
        return (self.be.sum(-t * self.logscale * self.be.safelog(y), axis=0))

    def bprop(self, y, t):
        """
        Computes the shortcut derivative of the multiclass cross entropy cost
        function

        Args:
            y (Tensor or OpTree): Output of previous layer or model
            t (Tensor or OpTree): True targets corresponding to y

        Returns:
            OpTree: Returns the (mean) shortcut derivative of the multiclass
            entropy cost function ``(y - t) / y.shape[1]``
        """
        return self.scale * (y - t)


class SumSquared(Cost):

    """
    Applies the squared error cost function
    """

    def __init__(self):
        """
        Initialize the squared error cost functions
        """
        self.func = lambda y, t: self.be.sum(
            self.be.square(y - t), axis=0) / 2.
        self.funcgrad = lambda y, t: (y - t)


class MeanSquared(Cost):

    """
    Applies the mean squared error cost function
    """

    def __init__(self):
        """
        Initialize the squared error cost functions
        """
        self.func = lambda y, t: self.be.mean(
            self.be.square(y - t), axis=0) / 2.
        self.funcgrad = lambda y, t: (y - t)/y.shape[0]



class TopKMisclassification(Metric):

    """
    Compute logloss, top1, and topk misclassification error metric
    """

    def __init__(self, k):
        self.correctProbs = self.be.iobuf(1)
        self.top1 = self.be.iobuf(1)
        self.topk = self.be.iobuf(1)

        self.k = k
        self.metric_names = ['LogLoss', 'Top1Misclass', 'Top' + str(k) + 'Misclass']

    def __call__(self, y, t, calcrange=slice(0, None)):
        """
        Compute the misclassification error metric

        Args:
            y (Tensor or OpTree): Output of previous layer or model
            t (Tensor or OpTree): True targets corresponding to y

        Returns:
            float: Returns the metric
        """
        be = self.be
        self.correctProbs[:] = be.sum(y * t, axis=0)
        nSlots = self.k - be.sum((y > self.correctProbs), axis=0)
        nEq = be.sum(y == self.correctProbs, axis=0)
        self.topk[:] = 1. - (nSlots > 0) * ((nEq <= nSlots) * (1 - nSlots / nEq) + nSlots / nEq)
        self.top1[:] = 1. - (be.max(y, axis=0) == self.correctProbs) / nEq
        self.correctProbs[:] = -be.safelog(self.correctProbs)
        return np.array((self.correctProbs.get()[:, calcrange].mean(),
                         self.top1.get()[:, calcrange].mean(),
                         self.topk.get()[:, calcrange].mean()))


class Misclassification(Metric):

    """
    Compute the misclassification error metric
    """

    def __init__(self):
        self.preds = self.be.iobuf(1)
        self.hyps = self.be.iobuf(1)
        self.outputs = self.preds  # Contains per record metric
        self.metric_names = ['Top1Misclass']

    def __call__(self, y, t, calcrange=slice(0, None)):
        """
        Compute the misclassification error metric

        Args:
            y (Tensor or OpTree): Output of previous layer or model
            t (Tensor or OpTree): True targets corresponding to y

        Returns:
            float: Returns the metric
        """
        # convert back from onehot and compare
        self.preds[:] = self.be.argmax(y, axis=0)
        self.hyps[:] = self.be.argmax(t, axis=0)
        self.outputs[:] = self.be.not_equal(self.preds, self.hyps)

        return self.outputs.get()[:, calcrange].mean()


class Accuracy(Metric):

    """
    Compute the accuracy metric
    """

    def __init__(self):
        self.preds = self.be.iobuf(1)
        self.hyps = self.be.iobuf(1)
        self.outputs = self.preds  # Contains per record metric
        self.metric_names = ['Accuracy']

    def __call__(self, y, t, calcrange=slice(0, None)):
        """
        Compute the accuracy metric

        Args:
            y (Tensor or OpTree): Output of previous layer or model
            t (Tensor or OpTree): True targets corresponding to y

        Returns:
            float: Returns the metric
        """
        # convert back from onehot and compare
        self.preds[:] = self.be.argmax(y, axis=0)
        self.hyps[:] = self.be.argmax(t, axis=0)
        self.outputs[:] = self.be.equal(self.preds, self.hyps)

        return self.outputs.get()[:, calcrange].mean()

class PrecisionRecallMetric(Metric):

    '''
    Compute precision and recall metrics
    '''

    def __init__(self, labels, time_steps, has_mask=False):

        """
        Create a precision and recall metric
        :param labels: (Array of Strings) the label names (in order)
        :param time_steps: number of time steps in the output layer
        :param has_mask: (boolean) this needs to be set to true
        :return:
        """

        self.outputs = self.be.zeros((2, len(labels)+1))
        self.token_stats = self.be.zeros((3, len(labels)+1))
        self.metric_names = ['Precision', 'Recall']
        self.yclass = self.be.iobuf(time_steps, dtype=np.int32).reshape((1, -1))
        self.time_steps = time_steps
        self.has_mask = has_mask

    def __call__(self, y, t):

        """
        Compute the precision and recall of a multi-class classification model

        Args:
            y (Tensor or OpTree): Output of previous layer or model
            t (Tensor or OpTree): True targets corresponding to y

        Returns:
            float: Returns the precision and recall values
        """

        self.yclass[:] = self.be.argmax(y, axis=0)
        y[:] = self.be.onehot(self.yclass, axis=0)

        if self.has_mask:
            t = t[0]

        # True positives
        self.token_stats[0, :] = self.be.sum(y*t, axis=1).T

        # Prediction
        self.token_stats[1, :] = self.be.sum(y, axis=1).T

        # Targets
        self.token_stats[2, :] = self.be.sum(t, axis=1).T

        # Precision
        self.outputs[0, :] = self.token_stats[0, :] / (self.token_stats[1, :]+0.0000001)

        # Recall
        self.outputs[1, :] = self.token_stats[0, :] / (self.token_stats[2, :]+0.0000001)

        # reshape here so there is no summing across the label predictions
        return self.outputs