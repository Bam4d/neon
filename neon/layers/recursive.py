__author__ = 'bam4d'

from neon.layers.layer import interpret_in_shape
import neon


def get_steps(x, shape):
    """
    Convert a (vocab_size, steps * batch_size) array
    into a [(vocab_size, batch_size)] * steps list of views
    """
    steps = shape[1]
    xs = x.reshape(shape + (-1,))
    return [xs[:, step, :] for step in range(steps)]

class Recursive(neon.layers.layer.ParameterLayer):

    """
    Basic recursive layer based on Recursive Deep Models for Semantic Compositionality
Over a Sentiment Treebank, Richard Socher, Alex Perelygin, Jean Y. Wu, Jason Chuang,
Christopher D. Manning, Andrew Y. Ng and Christopher Potts (2013)

    Arguments:
        output_size (int): Number of hidden/output units
        init (Initializer): Function for initializing the model parameters
        activation (Transform): Activation function for the input modulation

    Attributes:
        W_input (Tensor): weights from inputs to output units
            (input_size, output_size)
        W_recur (TTensor): weights for recurrent connections
            (output_size, output_size)
        b (Tensor): Biases on output units (output_size, 1)
    """

    def __init__(self, output_size, init, activation,
                 reset_cells=False, name="RecursiveLayer"):
        super(Recursive, self).__init__(init, name)
        self.x = None
        self.in_deltas = None
        self.nout = output_size
        self.h_nout = output_size
        self.activation = activation
        self.h_buffer = None
        self.W_input = None
        self.ngates = 1
        self.reset_cells = reset_cells

    def configure(self, in_obj):
        super(Recursive, self).configure(in_obj)
        (self.nin, self.nsteps) = self.in_shape
        self.out_shape = (self.nout, self.nsteps)
        self.gate_shape = (self.nout * self.ngates, self.nsteps)
        self.embeddings = self.be.array(in_obj.embeddings)

        if self.weight_shape is None:
            self.weight_shape = (self.nout, self.nin)
        return self

    def allocate(self, shared_outputs=None, shared_deltas=None):
        super(Recursive, self).allocate(shared_outputs, shared_deltas)

        # Embedding deltas
        self.embeddings_delta = self.be.zeros_like(self.embeddings)

        self.h_buffer = self.outputs
        self.out_deltas_buffer = self.deltas

        self.h = get_steps(self.h_buffer, self.out_shape)
        self.out_delta = get_steps(self.out_deltas_buffer, self.in_shape)

        # State deltas
        self.h_delta = get_steps(self.be.iobuf(self.out_shape), self.out_shape)
        self.bufs_to_reset = [self.h_buffer]

        if self.W_input is None:
            self.init_params(self.weight_shape)

    def get_xs_delta(self, code):
        code = code.split()
        idx = int(code[1])
        if code[0] is 'i':
            return self.embeddings_delta[:,idx]
        else:
            return self.h_delta[idx]

    def get_xs(self, code):
        code = code.split()
        idx = int(code[1])
        if code[0] is 'i':
            return self.embeddings[:,idx]
        else:
            return self.h[idx]

    def init_buffers(self, inputs):
        """
        Initialize buffers for recurrent internal units and outputs.
        Buffers are initialized as 2D tensors with second dimension being steps * batch_size
        A list of views are created on the buffer for easy manipulation of data
        related to a certain time step

        Arguments:
            inputs (Tensor): input data as 2D tensor. The dimension is
                             (input_size, sequence_length * batch_size)

        """
        if self.x is None or self.x is not inputs:
            if self.x is not None:
                for buf in self.bufs_to_reset:
                    buf[:] = 0

            self.xs_left_delta = []
            self.xs_right_delta = []
            self.xs_left = []
            self.xs_right = []
            self.x = inputs
            for i, (left, right) in enumerate(inputs):
                self.xs_left.append(self.get_xs(left))
                self.xs_left_delta.append(self.get_xs_delta(left))

                self.xs_right.append(self.get_xs(right))
                self.xs_right_delta.append(self.get_xs_delta(right))

    def init_params(self, shape):
        """
        Initialize params for LSTM including weights and biases.
        The weight matrix and bias matrix are concatenated from the weights
        for inputs and weights for recurrent inputs and bias.
        The shape of the weights are (number of inputs + number of outputs +1 )
        by (number of outputs * 4)

        Arguments:
            shape (Tuple): contains number of outputs and number of inputs

        """
        (nout, nin) = shape
        g_nout = self.ngates * nout
        # Weights: input, recurrent, bias
        if self.W is None:
            self.W = self.be.empty((nout + nin + 1, g_nout))
            self.dW = self.be.zeros_like(self.W)
            self.init.fill(self.W)
        else:
            # Deserialized weights and empty grad
            assert self.W.shape == (nout + nin + 1, g_nout)
            assert self.dW.shape == (nout + nin + 1, g_nout)

        self.W_input = self.W[:nin].reshape((g_nout, nin))
        self.b = self.W[-1:].reshape((g_nout, 1))

        self.dW_input = self.dW[:nin].reshape(self.W_input.shape)
        self.db = self.dW[-1:].reshape(self.b.shape)

    def fprop(self, inputs, inference=False):
        """
        Forward propagation of input to recursive layer.

        Arguments:
            inputs (Tensor): input to the model for each time step of
                             unrolling for each input in minibatch
                             shape: (vocab_size * steps, batch_size)
                             where:

                             * vocab_size: input size
                             * steps: degree of model unrolling
                             * batch_size: number of inputs in each mini-batch

            inference (bool, optional): Set to true if you are running
                                        inference (only care about forward
                                        propagation without associated backward
                                        propagation).  Default is False.

        Returns:
            Tensor: layer output activations for each time step of
                unrolling and for each input in the minibatch
                shape: (output_size * steps, batch_size)
        """
        self.init_buffers(inputs)

        if self.reset_cells:
            self.h[-1][:] = 0

        #test = self.be.zeros([1,1,1])
        #self.be.compound_dot(self.be.ones([1,100,1]), self.be.ones([100,1,1]), test, beta=1.0, alpha=1.0)

        for (h, xs_left, xs_right) in zip(self.h, self.xs_left, self.xs_right):
            self.be.compound_dot(self.W_input[:,:self.nin/2], xs_left, h)
            self.be.compound_dot(self.W_input[:,self.nin/2:], xs_right, h, beta=1.0)
            h[:] = self.activation(h + self.b)

        return self.h_buffer

    def bprop(self, deltas, do_acts=True):
        """
        Backward propagation of errors through recursive layer.

        Arguments:
            deltas (Tensor): tensors containing the errors for
                each step of model unrolling.
                shape: (output_size, * steps, batch_size)

        Returns:
            Tensor: back propagated errors for each step of time unrolling
                for each mini-batch element
                shape: (input_size * steps, batch_size)
        """
        self.dW[:] = 0

        if self.in_deltas is None:
            self.in_deltas = get_steps(deltas, self.out_shape)
            self.prev_in_deltas = self.in_deltas[-1:] + self.in_deltas[:-1]

        params = (self.xs_left, self.xs_right, self.h, self.h_delta,
                  self.in_deltas, self.xs_left_delta, self.xs_right_delta)

        for (xs_left, xs_right, hs, h_delta, in_deltas, xs_left_delta, xs_right_delta) \
                in reversed(zip(*params)):

            in_deltas[:] = self.activation.bprop(hs) * (in_deltas + h_delta)

            # Compute the delta weights for the left and the right inputs
            self.be.compound_dot(in_deltas, xs_left.T, self.dW_input[:,:self.nin/2], beta=1.0)
            self.be.compound_dot(in_deltas, xs_right.T, self.dW_input[:,self.nin/2:], beta=1.0)
            self.db[:] = self.db + self.be.sum(in_deltas, axis=1)

            # Compute the delta down for the left and the right inputs
            self.be.compound_dot(self.W_input[:,:self.nin/2].T, in_deltas, xs_left_delta)
            self.be.compound_dot(self.W_input[:,self.nin/2:].T, in_deltas, xs_right_delta)

        return self.embeddings_delta, self.h_delta

class TreeLSTM(Recursive):

    """
    The Tree LSTM model is based on Improved Semantic Representations From Tree-Structured Long Short-Term Memory Networks, Kai Sheng Tai, Richard Socher, Christopher D. Manning, ACL (2015) 

    Arguments:
        output_size (int): Number of hidden/output units
        init (Initializer): Function for initializing the model parameters
        activation (Transform): Activation function for the input modulation
        gate_activation (Transform): Activation function for the gates

    Attributes:
        x (Tensor): input data as 2D tensor. The dimension is
                    (input_size, sequence_length * batch_size)
        W_input (Tensor): Weights on the input units
            (out size * 4, input size)
        W_recur (Tensor): Weights on the recursive inputs
            (out size * 4, out size)
        b (Tensor): Biases (out size * 4 , 1)
    """
    def __init__(self, output_size, init, activation, gate_activation,
                 reset_cells=False, name="TreeLstmLayer"):
        super(TreeLSTM, self).__init__(output_size, init, activation, name)
        self.gate_activation = gate_activation
        self.ngates = 5  # Input, Output, Left forget, Right forget, Cell
        self.reset_cells = reset_cells

    def get_fs_delta(self, code):
        code = code.split()
        idx = int(code[1])
        if code[0] is 'i':
            return self.c_embeddings_delta[:,idx] # Really don't even need this to be honest...
        else:
            return self.f_delta[idx]

    def get_fs(self, code):
        code = code.split()
        idx = int(code[1])
        if code[0] is 'i':
            return self.be.zeros(self.nin/2)
        else:
            return self.f[idx]

    def get_cs_delta(self, code):
        code = code.split()
        idx = int(code[1])
        if code[0] is 'i':
            return self.c_embeddings_delta[:,idx] # Really don't even need this to be honest...
        else:
            return self.c_delta[idx]

    def get_cs(self, code):
        code = code.split()
        idx = int(code[1])
        if code[0] is 'i':
            return self.be.zeros(self.nin/2)
        else:
            return self.c[idx]

    def init_buffers(self, inputs):
        super(TreeLSTM, self).init_buffers(inputs)
        """
        Initialize buffers for recursive internal units

        Arguments:
            inputs (Tensor): input data as 2D tensor. The dimension is
                             (input_size, sequence_length * batch_size)

        """

        self.cs_left_delta = []
        self.cs_right_delta = []
        self.cs_left = []
        self.cs_right = []

        for i, (left, right) in enumerate(inputs):
            self.cs_left.append(self.get_cs(left))
            self.cs_left_delta.append(self.get_cs_delta(left))

            self.cs_right.append(self.get_cs(right))
            self.cs_right_delta.append(self.get_cs_delta(right))


    def allocate(self, shared_outputs=None, shared_deltas=None):
        super(TreeLSTM, self).allocate(shared_outputs, shared_deltas)

        # Not sure what this will be used for to be honest, but its the cell deltas that pass down to embeddings
        self.c_embeddings_delta = self.be.zeros_like(self.embeddings)

        # indices for slicing gate buffers
        (ifo1, ifo2) = (0, self.nout * 3)
        (i1, i2) = (0, self.nout)
        (f1, f2, f3) = (self.nout, self.nout * 2, self.nout * 3)
        (o1, o2) = (self.nout * 3, self.nout * 4)
        (g1, g2) = (self.nout * 4, self.nout * 5)

        # States: hidden, cell, previous hidden, previous cell
        self.c_buffer = self.be.iobuf(self.out_shape)
        self.c = get_steps(self.c_buffer, self.out_shape)

        self.c_act_buffer = self.be.iobuf(self.out_shape)
        self.c_act = get_steps(self.c_act_buffer, self.out_shape)

        # Gates: input, forget, output, input modulation
        self.ifog_buffer = self.be.iobuf(self.gate_shape)
        self.ifog = get_steps(self.ifog_buffer, self.gate_shape)
        self.ifo = [gate[ifo1:ifo2] for gate in self.ifog]
        self.i = [gate[i1:i2] for gate in self.ifog]

        # Left and right forget gates
        self.fs_left = [gate[f1:f2] for gate in self.ifog]
        self.fs_right = [gate[f2:f3] for gate in self.ifog]

        self.o = [gate[o1:o2] for gate in self.ifog]
        self.g = [gate[g1:g2] for gate in self.ifog]

        # State deltas
        self.c_delta_buffer = self.be.iobuf((self.out_shape))
        self.c_delta = get_steps(self.c_delta_buffer, self.out_shape)

        # Pre activation gate deltas
        self.ifog_delta_buffer = self.be.iobuf(self.gate_shape)
        self.ifog_delta = get_steps(self.ifog_delta_buffer, self.gate_shape)
        self.i_delta = [gate[i1:i2] for gate in self.ifog_delta]
        self.fs_left_delta = [gate[f1:f2] for gate in self.ifog_delta]
        self.fs_right_delta = [gate[f2:f3] for gate in self.ifog_delta]
        self.o_delta = [gate[o1:o2] for gate in self.ifog_delta]
        self.g_delta = [gate[g1:g2] for gate in self.ifog_delta]
        self.bufs_to_reset.append(self.c_buffer)

    def fprop(self, inputs, inference=False):
        """
        Apply the forward pass transformation to the input data.  The input
            data is a list of inputs with an element for each time step of
            model unrolling.

        Arguments:
            inputs (Tensor): input data as 2D tensors, then being converted into a
                             list of 2D slices

        Returns:
            Tensor: LSTM output for each model time step
        """
        self.init_buffers(inputs)

        if self.reset_cells:
            self.h[-1][:] = 0
            self.c[-1][:] = 0

        params = (self.h, self.c, self.xs_left, self.xs_right, self.ifog, self.ifo,
                  self.i, self.fs_left, self.fs_right, self.o, self.g, self.cs_left, self.cs_right,  self.c_act)

        for (h, c, xs_left, xs_right, ifog, ifo, i, fs_left, fs_right, o, g, cs_left, cs_right, c_act) in zip(*params):
            self.be.compound_dot(self.W_input[:,:self.nin/2], xs_left, ifog)
            self.be.compound_dot(self.W_input[:,self.nin/2:], xs_right, ifog, beta=1.0)
            ifog[:] = ifog + self.b

            ifo[:] = self.gate_activation(ifo)
            g[:] = self.activation(g)

            c[:] = fs_left * cs_left + fs_right * cs_right + i * g
            c_act[:] = self.activation(c)
            h[:] = o * c_act

        return self.h_buffer

    def bprop(self, deltas, do_acts=True):
        """
        Backpropagation of errors, output delta for previous layer, and
        calculate the update on model parmas

        Arguments:
            deltas (list[Tensor]): error tensors for each time step
                of unrolling
            do_acts (bool, optional): Carry out activations.  Defaults to True

        Attributes:
            dW_input (Tensor): input weight gradients

            db (Tensor): bias gradients


        Returns:
            Tensor: Backpropagated errors for each time step
                of model unrolling
        """
        self.c_delta_buffer[:] = 0
        self.dW[:] = 0

        if self.in_deltas is None:
            self.in_deltas = get_steps(deltas, self.out_shape)
            self.prev_in_deltas = self.in_deltas[-1:] + self.in_deltas[:-1]
            self.ifog_delta_last_steps = self.ifog_delta_buffer[:, self.be.bsz:]
            self.h_first_steps = self.h_buffer[:, :-self.be.bsz]

        params = (self.h_delta, self.in_deltas, self.xs_left_delta, self.xs_right_delta, self.prev_in_deltas,
                  self.i, self.fs_left, self.fs_right, self.o, self.g, self.ifog_delta,
                  self.i_delta, self.fs_left_delta, self.fs_right_delta, self.o_delta, self.g_delta,
                  self.c_delta, self.cs_left_delta, self.cs_right_delta, self.cs_left, self.cs_right,
                  self.c_act, self.xs_left, self.xs_right)

        for (h_delta, in_deltas, xs_left_delta, xs_right_delta, prev_in_deltas,
             i, f_left, f_right, o, g, ifog_delta, i_delta, f_left_delta, f_right_delta, o_delta, g_delta,
             c_delta, cs_left_delta, cs_right_delta, cs_left, cs_right, c_act, xs_left, xs_right) in reversed(zip(*params)):

            # current cell delta
            c_delta[:] = c_delta + self.activation.bprop(c_act) * (o * in_deltas)
            i_delta[:] = self.gate_activation.bprop(i) * c_delta * g
            f_left_delta[:] = self.gate_activation.bprop(f_left) * c_delta * cs_left
            f_right_delta[:] = self.gate_activation.bprop(f_right) * c_delta * cs_right
            o_delta[:] = self.gate_activation.bprop(o) * in_deltas * c_act
            g_delta[:] = self.activation.bprop(g) * c_delta * i

            # out deltas
            #self.be.compound_dot(self.W_input.T, ifog_delta, h_delta)
            self.be.compound_dot(self.W_input[:,:self.nin/2].T, ifog_delta, xs_left_delta)
            self.be.compound_dot(self.W_input[:,self.nin/2:].T, ifog_delta, xs_right_delta)

            cs_left_delta[:] = c_delta * f_left
            cs_right_delta[:] = c_delta * f_right

            prev_in_deltas[:] = prev_in_deltas + h_delta
            # sum and accumulate
            self.be.compound_dot(ifog_delta, xs_left.T, self.dW_input[:,:self.nin/2], beta=1.0)
            self.be.compound_dot(ifog_delta, xs_right.T, self.dW_input[:,:self.nin/2], beta=1.0)



        # Bias delta and accumulate
        self.db[:] = self.be.sum(self.ifog_delta_buffer, axis=1)


        return self.embeddings_delta, self.h_delta
