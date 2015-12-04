__author__ = 'bam4d'


import numpy as np
from neon.backends import gen_backend
from neon import NervanaObject

be = gen_backend(backend='cpu', batch_size=2, rng_seed=1234, device_id=0)

class NeuralStack(NervanaObject):

    def __init__(self, memory_steps_max=10, memory_vector_size=3):
        self._memory_steps_max = memory_steps_max
        self._memory_vector_size = memory_vector_size
        self._memory = self.be.zeros((self.be.bsz, memory_vector_size, memory_steps_max))
        self._strengths = self.be.iobuf(memory_steps_max)
        self._output = self.be.zeros((self.be.bsz, memory_vector_size ))
        # How many steps have occurred
        self._mem_pointer = 3

    def update_strengths(self, pop, push):
        assert pop.shape[0] == self.be.bsz
        assert push.shape[0] == self.be.bsz

        current_strengths = self._strengths[:self._mem_pointer]
        new_strengths = self._strengths[:self._mem_pointer+1]
        cs = self.be.iobuf(self._mem_pointer)
        cs[:-1] = np.cumsum(current_strengths[::-1][:-1].get(), axis=0)[::-1]

        new_strengths[:self._mem_pointer] = self.be.maximum(0, current_strengths-self.be.maximum(0, pop.T-cs))
        new_strengths[self._mem_pointer, :] = push
        self._mem_pointer += 1

    def get_output(self):
        current_memory = self._memory[:,:,:self._mem_pointer]
        current_strengths = self._strengths[:self._mem_pointer]
        cs = self.be.iobuf(self._mem_pointer)
        cs[:-1] = np.cumsum(current_strengths[::-1][:-1].get(), axis=0)[::-1]
        cs[:] = self.be.minimum(current_strengths, self.be.maximum(0, 1-cs))

        #TODO: How do I do this parallel?
        for b in range(0, self.be.bsz):
            self.be.compound_dot(cs[:,b].T, current_memory[b,:,:].T, self._output[b])

        return self._output


# Setup goes here
memory_size = 3
memory_steps = 10
stack = NeuralStack(memory_steps_max=memory_steps, memory_vector_size=memory_size)

# This is how many steps we have currently done
stack._mem_pointer = 3

stack._strengths[0] = 0.5
stack._strengths[1] = 0.4
stack._strengths[2] = 1.0

memory = np.zeros((2,3,3))

memory[0,:,:] = np.array([[1,0,0],[0,2,0],[0,0,3]])
memory[1,:,:] = np.array([[1,0,0],[0,2,0],[0,0,3]])

stack._memory[:, :, :3] = be.array(memory)

stack.update_strengths(be.array([0.0, 0.2]), be.array([0.0,0.0]))

print stack.get_output().get()

print stack._memory.get()
print stack._strengths.get()