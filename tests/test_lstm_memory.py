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
'''
Test of the LSTM with neural stack
'''

import itertools as itt
import numpy as np

from neon import NervanaObject
from neon.initializers.initializer import GlorotUniform
from neon.layers.recurrent import LSTM, get_steps
from neon.layers.memory import NeuralStack
from neon.transforms import Logistic, Tanh
from numpy import concatenate as con


def eq(candidate, expected):
    assert np.allclose(candidate, expected), "Tensors do not match, test failed"

def test_update_stack(backend_default):

    # Setup goes here
    be = NervanaObject.be
    be.bsz = 2

    memory_size = 3
    memory_steps = 10
    stack = NeuralStack(memory_steps_max=memory_steps, memory_vector_size=memory_size)

    # This is how many steps we have currently done
    stack._mem_pointer = 3
    m = stack._mem_pointer

    stack._strengths[:m, 0] = be.array(np.array([0.4,0.1,0.3]))
    stack._strengths[:m, 1] = be.array(np.array([0.4,0.1,0.3]))

    stack.update_strengths(be.array(np.array([0.0, 0.2])), be.array(np.array([0.0, 0.0])))

    eq(stack._strengths[:stack._mem_pointer-1].get(), np.array([[0.4, 0.1, 0.3], [0.4, 0.1, 0.1]]).T)

def test_get_output(backend_default):

    # Setup goes here
    be = NervanaObject.be
    be.bsz = 3

    memory_size = 3
    memory_steps = 10

    stack = NeuralStack(memory_steps_max=memory_steps, memory_vector_size=memory_size)

    # This is how many steps we have currently done
    stack._mem_pointer = 3
    m = stack._mem_pointer

    stack._strengths[:m, 0] = be.array(np.array([0.5,0.4,1.0]))
    stack._strengths[:m, 1] = be.array(np.array([0.5,0.4,0.8]))
    stack._strengths[:m, 2] = be.array(np.array([0.3,0.3,0.3]))

    stack._memory[0, :, :m] = be.array(np.array([[1,1,1],[2,2,2],[3,0,3]]))
    stack._memory[1, :, :m] = be.array(np.array([[1,1,1],[2,2,2],[3,0,3]]))
    stack._memory[2, :, :m] = be.array(np.array([[1,0,0],[0,2,0],[0,0,3]]))

    eq(stack.get_output().get(), np.array([[3.0, 0.0, 3.0], [2.8, 0.4, 2.8], [0.3, 0.6, 0.9]]))