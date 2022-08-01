def dot(A, B):
    '''Helper function that dots tensor A and tensor B
    Params: A: a regularly structured list. Irregular list is not allowed.
            B: a regularly structured list. Irregular list is not allowed.
    :returns out_array: output array resulting from A^T B i.e A.B
             shape : The desired shape of the output array. If none,
                     output array is already of the desired shape.
    '''
    out_array = None
    shape = None
    Ashape = get_shape(A)
    Bshape = get_shape(B)
    # either A or B is 0 dimensional i.e scalar
    if Ashape == [1] or Bshape == [1]:
        scalar = A[0] if Ashape == [1] else B[0]
        array = A if Bshape == [1] else B
        output_array = None
        shape = Ashape if array == A else Bshape
    # A and B are 1 D
    elif len(Ashape) == 1 and len(Bshape) == 1:
        if Ashape[0] != Bshape[0]:
            raise ValueError(f"Shape {Ashape} is not compatible with {Bshape}")
        out_array = [sum(i * j for (i, j) in zip(A, B))]
        shape = [1]
    # both A and B are 2 dimensional
    elif len(Ashape) == 2 and len(Bshape) == 2:
        # check shape
        if Ashape[-1] != Bshape[-2]:
            raise ValueError(f"Shape {Ashape} is not compatible with {Bshape}")
        out_array = [[sum([re * ce for re, ce in zip(row_A, col_B)]) for col_B in zip(*B)] for row_A in A]
        shape = None
    # both A and B and 2+ dimensional


    return out_array, shape

def get_shape(lst, shapelst=()):
    '''Returns the list shape of the given lst
    :param lst: a regularly structured list. Staggered lists are not allowed.
    :returns output_list: a list of the shape of lst e.g [1, 2] for array: [[1, 2]]'''
    def helper(lst, shapelst=()):
        if not isinstance(lst, list):
            return shapelst
        if isinstance(lst[0], list):
            innerlen = len(lst[0])
            if not all(len(item) == innerlen for item in lst):
                raise ValueError("All lists dont have same length")
        shapelst += (len(lst),)
        shapelst = helper(lst[0], shapelst)
        return shapelst
    return list(helper(lst))

def flatten(array):
    '''Flattens the array into a 1D array'''
    if len(array) == 0:
        return array
    if type(array[0]) is list:
        return flatten(array[0]) + flatten(array[1:])
    return [array[0]] + flatten(array[1:])
