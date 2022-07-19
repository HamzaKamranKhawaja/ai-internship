from Tensor import Tensor
from Helper import dot

A = Tensor([4], [1])
B = Tensor([1, 2, 3, 4], [2, 2])

print(dot(A, B).data)