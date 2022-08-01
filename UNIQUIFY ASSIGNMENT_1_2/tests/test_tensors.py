import unittest
from Tensor import Tensor
from Helper import *
import io
import sys
import numpy as np
from Dense import Dense
import tensorflow as tf
from Activations import *

from keras import backend as K

class TestTensor(unittest.TestCase):

    def test_shape_edgecases(self):
        """Tests cases where shape is not as intended i.e empty, has non-positive integers, or non-integers"""
        print("Testing shape edge cases...")
        self.assertEqual(Tensor([0, 1], [0]).reshaped, [], "Reshaped should be [] if 0 values in shape")
        self.assertEqual(Tensor([0, 1], [-1]).reshaped, [], "Reshaped should be [] if negative values in shape")


        #TODO: CAPTURE STRING OUTPUT ON CONSOLE AND COMPARE TO ERROR MESSAGE.
        #testing specific error scenarios that print to output
        # output = io.StringIO()
        # sys.stdout = output
        # Tensor([1, 2, 3], ["ab"])
        # sys.stdout = sys.__stdout__
        # self.assertEqual(output.getvalue(), "shape has a non integer value\n")
        #
        # output = io.StringIO()
        # sys.stdout = output
        # Tensor([1, 2, 3], [-1])
        # sys.stdout = sys.__stdout__
        # self.assertEqual(output.getvalue(), "shape has non positive integer\n")
        # print("shape tests passed.")

    def test_zeros(self):
        """Tests underlying substructure of zeros i.e the zero array which is initially created"""
        print("Testing zeros in Tensor...")
        i, j, k = 2, 3, 4
        self.assertEqual(Tensor([0,0,0], [3, 2]).reshaped, [[0] * i]*j, "Should be: " + str([[0] * i]*j))
        self.assertEqual(Tensor([0, 0, 0], [4, 3, 2]).reshaped, [[[0] * i] * j]*k, "Should be: " + str([[[0] * i] * j]*k))
        print("zeros passed.")


    def test_Tensor(self):
        """Tests Tensor class as a whole. Given tests in email incorporated"""
        print("Tensor Tests...")
        #general test
        #TODO: ADD TWO MORE TEST CASES
        self.assertEqual(Tensor([i for i in range(10)], [1, 10]).reshaped, [[i for i in range(10)]], "Should be: " + str([[i for i in range(10)]]))

        # the given two cases
        tensor = Tensor([0, 1, 2, 3, 4, 5, 0.1, 0.2, -3], [2, 3, 2])
        result = [[[0, 1], [2, 3], [4, 5]], [[0.1, 0.2], [-3, 0], [0, 0]]]
        self.assertEqual(tensor.reshaped, result, "Should be: " + str(result))

        tensor = Tensor([0, 1, 2, 3, 4, 5, 0.1, 0.2, -3, -2, -1, 3, 2, 1], [5, 2])
        result = [[0, 1], [2, 3], [4, 5], [0.1, 0.2], [-3, -2]]
        self.assertEqual(tensor.reshaped, result, "Should be: " + str(result))
        print("Tensor tests passed.")

    def test_Tensorfromarray(self):
        print("Testing tensor from Array. ")
        print("Checking Get shape function...")
        expected = [[1, 2, 3], [5, 5, 5], [1, 1, 1], [1, 1, 10]]
        for lst in expected:
            A = Tensor([1, 2, 3, 4, 5, 6, 7, 8, 9], lst)
            shape = get_shape(A.reshaped)
            self.assertEqual(shape, lst, f"Shape {shape} should equal {lst}")
        print("Get shape passed.")

        print("Testing flatten...")
        lsts = [([1, 2, 3], [[1], [2], [3]]), ([1, 1, 1, 1], [[1, 1], [1, 1]]), ([2, 1, 2, 1], [[[[2, 1, 2, 1]]]])]
        for tup in lsts:
            expected = tup[0]
            flattened = flatten(tup[1])
            self.assertEqual(expected, flattened, f"{flattened} should equal {expected}")
        print("flatten passed.")

        print("Testing Tensor from Array creation...")
        arrays = [([1, 2, 3], [3], [1, 2, 3]), ([[1, 2], [3, 4]], [2, 2], [1, 2, 3, 4]), ([[[1, 2]]], [1, 1, 2], [1, 2])]
        for tup in arrays:
            array = tup[0]
            shape = tup[1]
            data = tup[2]
            A = Tensor(array)
            self.assertEqual(A.reshaped, array, f"{A.reshaped} should equal {array}")
            self.assertEqual(A.shape, shape, f"{A.shape} should equal {shape}")
            self.assertEqual(A.data, data, f"{A.data} should equal {data}")
        print("Tensor from array passed.")

    def test_Tensordot(self):
        print("Tensor dot testing...")
        arrays1 = [[1, 1, 1, 1], [[1, 2], [3, 4]], [5], [[1, 1], [1, 1]]]
        arrays2 = [[4, 4, 4, 4], [[1, 2], [3, 4]], [[2, 2], [2, 2]] , [[2], [2]]]
        results = [[16],       [[7, 10], [15, 22]], [[10, 10], [10, 10]], [[4], [4]]]

        #check first output of prod, which is the array. Second ignored output will be shape of that array
        for arr1, arr2, r in zip(arrays1, arrays2, results):
            A = Tensor(arr1)
            B = Tensor(arr2)
            C = Tensor.dot(A, B)
            self.assertEqual(C.reshaped, r, f"{arr1} @ {arr2} should equal {r}, not {C.reshaped}")
        print("Tensor dot passed.")

    def test_Tensoradd(self):
        print("Testing tensor ADD...")
        array1 = [[1, 2, 3, 4], [[1, 2], [3, 4]], [5], [[[[1]]]]]
        array2 = [[1, 1, 1, 1], [[1, 0], [0, 1]], [6], [[[[8]]]]]
        results = [[2, 3, 4, 5], [[2, 2], [3, 5]], [11], [[[[9]]]]]
        for a1, a2, r in zip(array1, array2, results):
            A = Tensor(a1)
            B = Tensor(a2)
            C = Tensor.add(A, B)
            self.assertEqual(C.reshaped, r, f"{a1} + {a2} should equal {r}, not {C.reshaped}")
        print("ADD passed.")

    def test_Dense(self):
        print("Testing Dense layers...")
        w = Tensor([[1, 1], [1, 1], [1, 1]])
        x = Tensor([1, 1, 1])
        print(w.shape[-1], x.shape[0])
        b = Tensor([1, 0])
        dense = Dense(w, b)
        result = dense.forward(x).reshaped
        self.assertEqual(result, [4, 3])

        w = Tensor([[1, 6, 5], [3, 4, 8], [2, 12, 3]])
        x = Tensor([[3, 4, 6], [5, 6, 7], [6, 56, 7]])
        b = Tensor([[0]*3]*3)
        dense = Dense(w, b)
        result = dense.forward(x).reshaped
        self.assertEqual(result, [[27, 106, 65], [37, 138, 94], [188, 344, 499]])

        print("Importing model for testing...")
        model = tf.keras.models.load_model('./../assignment1-model2')
        print(model.summary())
        #compare model data with your own, use weights + biases etc
        print(f"number of layers: {len(model.layers)}")


        # layer 0: flatten from (28 x 28) -> (784, 1)
        # layer 1: (784, 1) -> (128, 1)
        # layer 2:  (128, 1) -> (48, 1)
        # layer 3:  (48, 1) -> (10, 1)

        numofpredictions = 150

        #get data
        mnist_dataset = tf.keras.datasets.mnist
        (_, _), (x_test, y_test) = mnist_dataset.load_data()

        #make predictions using model
        all_model_predictions = np.argmax(model.predict(x_test[:numofpredictions]), axis=1)
        print(all_model_predictions)
        all_our_predictions = []
        for i in range(numofpredictions):
            arr = x_test[i]

            # if layer outputs are needed
            # inp = model.input                                           # input placeholder
            # outputs = [layer.output for layer in model.layers]          # all layer outputs
            # functors = [K.function([inp], [out]) for out in outputs]    # evaluation functions
            # layer_outs = [func([inp]) for func in functors]
            # print(layer_outs)


            #obtain weights and biases for all layers
            weight_biases = [layer.get_weights() for layer in model.layers if len(layer.get_weights()) > 0]

            #flatten initial input
            next_in = Tensor(flatten(arr.tolist()))

            #numpy array to check whether forward pass is working properly
            arr = np.array(flatten(arr.tolist()))

            #after each layer, check that the numpy array is equal to our array
            for weight, bias in weight_biases:
                arr = arr @ weight + bias
                weight, bias = weight.tolist(), bias.tolist()
                A = Tensor(weight)
                B = Tensor(bias)
                curr_layer = Dense(A, B)
                next_in = curr_layer.forward(next_in)
                next_in.apply(relu)

            our_prediction = np.argmax(next_in.reshaped, axis=0)
            all_our_predictions.append(our_prediction)

            #check if final prediction is the same as the model's prediction
            # print(f"model predicted: {digit_predicted}")
            # print(f"our prediction: {our_prediction}")
            # self.assertEqual(our_prediction, digit_predicted)
        print(all_our_predictions)
        self.assertEqual(all_model_predictions.tolist(), all_our_predictions)
if __name__ == "__main__":
    print("Running test suite.")
    unittest.main()
    print("All tests completed")

