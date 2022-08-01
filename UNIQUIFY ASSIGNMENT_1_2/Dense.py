from Tensor import Tensor


class Dense:
    """ Class Dense  to create a fully connected Dense layer"""

    def __init__(self, weights, biases):
        """ creates a Dense Object with tensors WEIGHTS, BIASES from tensor
        class
        :param weights: Tensor of weights, with size (INPUT LAYER DIMENSION, OUTPUT LAYER DIMENSION)
               biases: Tensor of biases, with size (OUTPUT LAYER DIMNESION)
        """
        self.weights = weights
        self.biases = biases
        self.output = None


    def forward(self, inputs):
        """ Steps the inputs tensor one step forward with forward propagation.
        XW + B where W = Weights, X = Inputs, B = Biases.
        XW = X.W_[i, j, k, m] = sum(X_[i, j, :] * W_[k, :, m])
        """
        #CHANGE FORWARD TO BE XW WHERE X = [[first input...], [second input...]...] and W = [[WEIG
        result = Tensor.dot(inputs, self.weights)
        if self.biases is not None:
            result = Tensor.add(result, self.biases)
        self.output = result
        return self.output

