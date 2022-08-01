import math
import copy
import Helper
from inspect import signature
from Helper import *

class Tensor:

    def __init__(self, data, shape=None):
        if shape:
            self.data = data
            self.shape = shape
            self.reshaped = []

            if self.check_shape():
                self.reshaped = self.reshape(data, shape)
        if not shape:
            self.data = flatten(data)
            self.shape = get_shape(data)
            self.reshaped = data

    def check_shape(self):
        """Checks shape is an iterable of positive integers"""

        for item in self.shape:
            try:
                if int(item) == item and item > 0:
                    continue
                else:
                    print("shape has non positive integer")
                    return False
            except ValueError:
                print("shape has a non integer value")
                return False
        # ensure shape is not an empty list and does not contain 0's
        shape_len = math.prod(self.shape)
        if len(self.shape) == 0 or not shape_len:
            print("Shape is not valid")
            return False
        return True


    def create_zeros(self):
        """Creates and returns substructure of 0's according to shape given, data is superimposed onto it"""
        shape = self.shape
        length = len(shape) - 1
        zeros = 0

        for i in range(length, -1, -1):
            dim = shape[i]
            final = []
            for _ in range(dim):
                if zeros:
                    temp = copy.deepcopy(zeros)
                else:
                    temp = 0
                final.append(temp)
            zeros = final[:]
        return zeros

    def reshape(self, data, shape):
        """Reshapes given data according to the nesting structure provided in shape."""

        final = self.create_zeros()
        # obtain the maximum index for iteration -- minimum of shape_len and data size
        shape_len = math.prod(shape)
        data_len = len(data)
        maxi = min(shape_len, data_len)
        indices = [0] * len(shape)

        if not shape_len:
            print("Shape is not valid")
            return []

        for i in range(maxi):
            lst = final
            for j in indices[:-1]:
                lst = lst[j]

            lst[indices[-1]] = data[i]
            indices = self.increment(indices, shape)

        return final

    def increment(self, indices, ceilings):
        """Increments the list of indices given by one, starting from the last entry.
        Ceilings provides the limits of increment"""

        ceilings = ceilings[::-1]
        temp = list(indices)[::-1]
        cont = True
        ind = 0
        while cont and ind < len(indices):
            cont = False
            if temp[ind] + 1 >= ceilings[ind]:
                cont = True
                temp[ind] = 0
            else:
                temp[ind] += 1
            ind += 1
        return temp[::-1]


    def apply(self, func):
        """Applies the given function to all elements in self.
        :param func: The function to apply. Can use functions in Activations.py for activation functions
        """
        data = [func(x) for x in self.data]
        self.reshaped = self.reshape(data, self.shape)
        self.data = data








    @classmethod
    def dot(cls, tensorA, tensorB):
        '''Helper function that dots tensor A and tensor B
        Params: A: a tensor object.
                B: a tensor object.
        :returns out_tensor: output tensor resulting from A^T B i.e A.B
        '''
        out_tensor = None
        Ashape = tensorA.shape
        Bshape = tensorB.shape
        A = tensorA.reshaped
        B = tensorB.reshaped
        dataA = tensorA.data
        dataB = tensorB.data
        # either A or B is 0 dimensional i.e scalar
        # we use data of tensor to multiply, then construct a new tensor
        if Ashape == [1] or Bshape == [1]:
            scalar = A[0] if Ashape == [1] else B[0]
            array = dataA if Bshape == [1] else dataB
            shape = Ashape if array == dataA else Bshape
            out_array = [scalar * val for val in array]
            out_tensor = Tensor(out_array, shape)
        # A and B are 1 D
        elif len(Ashape) == 1 and len(Bshape) == 1:
            if Ashape[0] != Bshape[0]:
                raise ValueError(f"Shape {Ashape} is not compatible with {Bshape}")
            out_array = [sum(i * j for (i, j) in zip(dataA, dataB))]
            out_tensor = Tensor(out_array)
        #Only 1 of the two is 1D:
        elif (len(Ashape) == 1 or len(Bshape) == 1) and (len(Ashape) <= 2 and len(Bshape) <= 2):
            #A matrix, B vector
            if len(Ashape) == 2:
                if Ashape[-1] != Bshape[0]:
                    raise ValueError(f"Shape {Ashape} is not compatible with {Bshape}")
                out_array = [[sum(re * ce for re, ce in zip(A_row, dataB))]for A_row in A]
                out_tensor = Tensor(out_array)
            #A vector, B matrix
            if len(Bshape) == 2:
                if Ashape[0] != Bshape[-2]:
                    raise ValueError(f"Shape {Ashape} is not compatible with {Bshape}")
                out_array = [sum(re * ce for (re, ce) in zip(dataA, col_B)) for col_B in zip(*B)]
                out_tensor = Tensor(out_array)
        #if Ashape[0] != Bshape[0]:
        #    raise ValueError(f"Shape {Ashape} is not compatible with {Bshape}")
        # both A and B are 2 dimensional
        elif len(Ashape) == 2 and len(Bshape) == 2:
            # check shape
            if Ashape[-1] != Bshape[-2]:
                raise ValueError(f"Shape {Ashape} is not compatible with {Bshape}")
            out_array = [[sum([re * ce for re, ce in zip(row_A, col_B)]) for col_B in zip(*B)] for row_A in A]
            out_tensor = Tensor(out_array)
        # one or both are 2+ dimensional
        #XW = X.W_[i, j, k, m] = sum(X_[i, j, :] * W_[k, :, m])
        else:
            rarray = []
            if len(Ashape) > 2:
               for i in range(Ashape[0]):
                    intermediate_tensor = Tensor(A[i]).dot(tensorB)
                    intermediate_array = intermediate_tensor.reshaped
                    rarray.append(intermediate_array)

            elif len(Bshape) > 2:
                for j in range(Bshape[0]):
                    intermediate_tensor = tensorA.dot(Tensor(B[j]))
                    intermediate_array = intermediate_tensor.reshaped
                    rarray.append(intermediate_array)

            out_tensor = Tensor(rarray)

        return out_tensor

    @classmethod
    def add(cls, A, B):
        '''Adds Tensor A and Tensor B to return a new Tensor
        :param A: Tensor
               B: Tensor
        :returns C: Tensor after adding Tensor A and Tensor B. C = A + B'''
        if A.shape != B.shape:
            raise ValueError(f"shape {A.shape} != {B.shape}")
        c_data = [a + b for (a, b) in zip(A.data, B.data)]
        c = Tensor(c_data, A.shape)
        return c

