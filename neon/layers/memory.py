__author__ = 'bam4d'


import numpy as np
from neon import NervanaObject


class NeuralStack(NervanaObject):

    def __init__(self, memory_steps_max=10, memory_vector_size=3):
        super(NeuralStack, self).__init__("Neural Stack")
        self._memory_steps_max = memory_steps_max
        self._memory_vector_size = memory_vector_size
        self._memory = self.be.zeros((self.be.bsz, memory_vector_size, memory_steps_max))
        self._strengths = self.be.iobuf(memory_steps_max)
        self._output = self.be.zeros((self.be.bsz, memory_vector_size ))

    def update_strengths(self, pop, push):
        assert pop.shape[0] == self.be.bsz
        assert push.shape[0] == self.be.bsz

        current_strengths = self._strengths[:self._mem_pointer]
        new_strengths = self._strengths[:self._mem_pointer+1]
        cs = self.be.iobuf(self._mem_pointer)

        # self.be.array(np.array( is a nasty hack for a strange underlying backend bug here...)
        cs[:-1] = self.be.array(np.array(np.cumsum(current_strengths[::-1][:-1].get(), axis=0)[::-1]))

        new_strengths[:self._mem_pointer] = self.be.maximum(0, current_strengths-self.be.maximum(0, pop.T-cs))
        new_strengths[self._mem_pointer, :] = push
        self._mem_pointer += 1

    def get_output(self):
        current_memory = self._memory[:, :, :self._mem_pointer]
        current_strengths = self._strengths[:self._mem_pointer]
        cs = self.be.iobuf(self._mem_pointer)

        # self.be.array(np.array( is a nasty hack for a strange underlying backend bug here...)
        cs[:-1] = self.be.array(np.array(np.cumsum(current_strengths[::-1][:-1].get(), axis=0)[::-1]))
        cs[:] = self.be.minimum(current_strengths, self.be.maximum(0, 1-cs))

        #TODO: How could this be parallel?
        for b in range(0, self.be.bsz):
            self.be.compound_dot(cs[:,b].T, current_memory[b,:,:], self._output[b])

        return self._output
