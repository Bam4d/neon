# ----------------------------------------------------------------------------
# Copyright 2015 Nervana Systems Inc.
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

from neon.layers.recurrent import LSTM
from neon.layers.layer import Affine
from neon.backends import gen_backend
from neon.initializers import Constant
from neon.models import Model
from neon.layers import GeneralizedCostMask
from neon.optimizers import GradientDescentMomentum

from neon.transforms import Logistic, SumSquared
from neon.callbacks.callbacks import Callbacks

from neon.data.tree import TreeDataset
from neon.layers.recursive import TreeLSTM
import numpy as np

import unittest
from unittest import TestCase
from neon.util.argparser import NeonArgparser

# parse the command line arguments
parser = NeonArgparser(__doc__)
args = parser.parse_args()

def numeric_gradient(model, inputs):

    delta = 0.01
    outputs = model.fprop(inputs)
    model.layers[0].embeddings += delta
    outputs_delta = model.fprop(inputs)

    return (outputs_delta - outputs) / delta

class RNNTests(TestCase):

    def setUp(self):
        wordVectorSize = 1
        self.nouts = 1

        # setup backend
        self.be = gen_backend(backend=args.backend,
                 batch_size=1,
                 rng_seed=1234,
                 device_id=args.device_id,
                 default_dtype=args.datatype)

        # Weight initialization all to 0.1 so we can test this against the recursive network
        init = Constant(val=0.1)
        #init = Uniform(low=-0.08, high=0.08)

        self.cost = GeneralizedCostMask(costfunc=SumSquared())

        self.optimizer = GradientDescentMomentum(0.01, 0)

        # Really simple vocab embedding set
        vocab = []
        vocab.append(np.ones([1,wordVectorSize])*0.1)
        vocab.append(np.ones([1,wordVectorSize])*0.15)
        vocab.append(np.ones([1,wordVectorSize])*0.2)
        vocab.append(np.ones([1,wordVectorSize])*0.25)

        self.targets_mask = (
            self.be.array(np.concatenate([
                np.zeros([1, self.nouts]),
                np.zeros([1 ,self.nouts]),
                np.ones([1, self.nouts])*0.5])).T,
            self.be.array(np.concatenate([
                np.zeros([1, self.nouts]),
                np.zeros([1, self.nouts]),
                np.ones([1, self.nouts])])
            ).T
        )

        self.embeddings = np.zeros([wordVectorSize, 4], dtype=np.float32)

        for i ,k in enumerate(vocab):
            self.embeddings[:,i] = np.array(k)

        # TODO: think of a better way of indexing this stuff, because currently this is a bit hacky and it does not make me
        # happy
        # for a binary tree the left and right vectors are combined in the dataset representation in the following way
        # embedding index 0 - embedding index 1 => ['i 0','i 1']
        # embedding index 2 - hidden index 0 => ['i 2','h 0']
        #
        # this gives us a binary tree that looks like the following:
        #
        #
        #           h 2
        #          /   \
        #        h 1   i 3
        #       /   \
        #     h 0   i 2
        #    /   \
        #  i 0   i 1
        #
        #
        # The order that the tree is traversed is given by the instruction list:
        #   [['i 0','i 1'], ['h 0','i 2']]
        #
        # As the tree is traversed, these instructions are converted into the tree node outputs

        self.tree_nodes = []
        self.tree_nodes.append(([['i 0', 'i 1'], ['h 0', 'i 2'], ['h 1', 'i 3']], self.targets_mask))
        self.tree_nodes.append(([['i 3', 'i 3'], ['h 0', 'i 3'], ['h 1', 'i 3']], self.targets_mask))
        self.tree_nodes.append(([['i 2', 'i 1'], ['h 0', 'i 2'], ['h 1', 'i 1']], self.targets_mask))

        self.dataset = TreeDataset(self.tree_nodes, 1, self.embeddings)

        self.recurrent_model = Model(layers=[
            LSTM(wordVectorSize, init, activation=Logistic(), gate_activation=Logistic()),
            Affine(self.nouts, init, bias=init, activation=Logistic())
        ])

        self.recursive_model = Model(layers=[
            TreeLSTM(wordVectorSize, init, activation=Logistic(), gate_activation=Logistic()),
            Affine(self.nouts, init, bias=init, activation=Logistic())
        ])

    def test_forward(self):
        '''
        Test that the tree forward pass is identical to the recurrent forward pass
        In this case the tree we are passing to the recursive model is a left-aligned tree
        :return:
        '''

        self.recurrent_model.initialize(self.embeddings)
        self.recursive_model.initialize(self.dataset)

        #create the list of steps for the recurrent net
        #out_recurrent = recurrent_model.fprop(be.array(embeddings))
        #print out_recurrent.get()

        out_recursive = self.recursive_model.fprop(self.tree_nodes[0][0])
        #print out_recursive.get()

        self.assertAlmostEqual(out_recursive[:,0].get(), 0.52675, 4)
        self.assertAlmostEqual(out_recursive[:,1].get(), 0.52690, 4)
        self.assertAlmostEqual(out_recursive[:,2].get(), 0.52705, 4)

    def test_backward(self):
        '''
        Test that the tree backwards pass is identical to the recurrent backward pass
        :return:
        '''
        self.recursive_model.initialize(self.dataset)

        x = self.recursive_model.fprop(self.tree_nodes[0][0])

        deltas = self.cost.get_errors(x, self.targets_mask)

        self.assertAlmostEqual(deltas[:,0].get(), 0.0, 4)
        self.assertAlmostEqual(deltas[:,1].get(), 0.0, 4)
        self.assertAlmostEqual(deltas[:,2].get(), 0.027050, 4)

        (embeddings_delta, h_delta) = self.recursive_model.bprop(deltas)

        self.assertAlmostEqual(embeddings_delta[:,0].get(), 0.00000019687, 10)
        self.assertAlmostEqual(embeddings_delta[:,1].get(), 0.00000019687, 10)
        self.assertAlmostEqual(embeddings_delta[:,2].get(), 0.00000037424, 10)
        self.assertAlmostEqual(embeddings_delta[:,3].get(), 0.00000563978, 10)

        self.assertAlmostEqual(h_delta[0], 0.00000037424, 10)
        self.assertAlmostEqual(h_delta[1], 0.00000563978, 10)
        self.assertAlmostEqual(h_delta[2], 0, 10)

    def test_train_epochs(self):

        callbacks = Callbacks(self.recursive_model, self.dataset)

        self.recursive_model.fit(self.dataset, self.cost, self.optimizer, 100, callbacks)

if __name__ == '__main__':
    unittest.main()