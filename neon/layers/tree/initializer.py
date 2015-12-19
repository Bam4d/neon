__author__ = 'bam4d'

from neon.initializers.initializer import Initializer


class PreTrained(Initializer):
    """
    A class for initializing parameter tensors with a matrix of pre-trained values.
    for example word2vec

    Args:
        val (np.array, optional): The array to assign the tensors
    """
    def __init__(self, array, name="pre-trained"):
        super(PreTrained, self).__init__(name=name)
        self.array = array

    def fill(self, param):
        param.fill(self.be.array(self.array).T)