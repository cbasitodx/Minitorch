from minitorch.Autograd import Value
from typing import List, Tuple, Any

class Tensor:

    """
    ===========
    **Summary**
    ===========

        A :class:`Tensor` is a multi-dimensional matrix containing elements of a single data type.

        This class only allows a maximum of four (4) dimensions. These can be seen as:

        * **0 dimensional tensor**: 
            Is a **scalar value**. It's initialized with a single element.
            
        * **1 dimensional tensor**:
            Is a **vector**. It's initialized with a list.

        * **2 dimensional tensor**:
            Is a **matrix**. It's initialized with a list of lists.
        
        * **3 dimensional tensor** & **4 dimensional tensor**:
            Is called a **tensor** in mathematics. It's initialized with a 3-dimensional or 4-dimensional list.
            The 3-dimensional tensor is useful for images that contains various channels (e.g a color image), while the 4-dimensional tensor is useful for representing **batches of data**, being the fourth dimension the batch size.
         
        .. warning::

            Uniformity along all dimensions is assumed. This means that, for example, the matrix [[1,2], [3]] is **not** supported as a tensor.
            Not mantaining uniformity may lead to undefined behaviour.
            
    ==============
    **Parameters**
    ==============
    
        * **shape:** (*Tuple[int]*) Dimensions of the tensor. Its read as (Batch Size, Number of Channels, Rows, Columns)

        * **__data:** (:class:`Autograd.Value` | *List*) List that contains the raw data of the tensor as Value objects.

    ======================
    **Instance Variables**
    ======================

        * **data:** (*int | float | Value | List | List[List] | List[List[List] | List[List[List[List]]]*) List containing the raw data of the tensor.

    ===========
    **Example**
    ===========

        >>> 3d_tens = Tensor([ [[1,2], [3,4], [4,5]], [[5,6], [7,8]] ])
        >>> scalar = Tensor(3.14)

    """
        
    def __init__(self, 
                 data : int | float | Value | List | List[List] | List[List[List]] | List[List[List[List]]]):

        # Empty lists are not allowed as instance variables
        if isinstance(data, list) and len(data) == 0:
            raise Exception("Error! Empty lists are invalid instance variables for this class")

        self.shape : Tuple[int]
        
        def get_dims(data : List):
            if type(data) != list:
                return []
            else:
                return [len(data)] + get_dims(data[0]) 
        
        self.shape = tuple(get_dims(data))

        if len(self.shape) > 4:
            raise Exception("Error! Tensors with more than four (4) dimensions are not supported")

        self.__data : Value | List

        # Converts the list (or list of lists or ...) of integers or floats to a list of Value objects
        def get_value_list(data : List) -> Value | List:
            if isinstance(data, list):
                return [get_value_list(item) for item in data]
            elif isinstance(data, (float,int)): 
                return Value(data)
            elif isinstance(data, Value):
                return data
            else:
                raise Exception("Error! Unable to wrap the raw data values inside Value objects")
        
        self.__data = get_value_list(data)
        
    def __iter__(self):
        return iter(self.__data)
    
    def __mul__(self, other : 'Tensor') -> 'Tensor':

        """
            In version 0.0.1, this class only supports scalar multiplication and vector or matrix multiplication.
            It is mathematically accurate.
        """

        res : Tensor

        # Scalar multiplication
        if len(self.shape) == 0:
            res = Tensor(self.__data * other.__data)
        
        # Vector or matrix multiplication
        else:
            # Check if self.rows == other.rows and if the tensors dont have more than 2 dimensions
            if self.shape[-1] != other.shape[-2] or len(self.shape) > 2 or len(other.shape) > 2:
                raise Exception("Error! Multiplication between Tensor objects can only be performed between matrices (2 dimensional Tensor), \
                                 vectors (1 dimensional Tensor) or scalars (0 dimensional Tensor)")
            
            self_rows = 1 if len(self.shape) == 1 else self.shape[0]

            res = Tensor([[Value(0) for _ in range(other.shape[1])]for _ in range(self_rows)])

            for self_row in range(self_rows):
                for other_col in range(other.shape[1]):
                    for other_row in range(other.shape[0]):

                        if len(self.shape) == 1:
                            res.__data[self_row][other_col] += self.__data[other_row] * other.__data[other_row][other_col]
                        else:
                            res.__data[self_row][other_col] += self.__data[self_row][other_row] * other.__data[other_row][other_col]

            # NOTE: This is done for dropping dimensions (flattening), e.g: [[1]] -> [1]
            while len(res.shape) != 1 and res.shape[0] == 1:
                res = res[0]

        return res
    
    def __add__(self, other : 'Tensor') -> 'Tensor':
        
        # Check if the tensors have the same dimentions and that they are, at most, 2-dimensional
        if self.shape != other.shape or len(self.shape) > 2 or len(other.shape) > 2:
            raise Exception("Error! Addition is only defined between tensors of the same dimention, and is not supported (as for v.0.0.1) for tensors that have more than 2 dimensions")

        res : Tensor

        # Scalar addition
        if len(self.shape) == 0:
            res = Tensor(self.__data + other.__data)

        # Vector addition
        if len(self.shape) == 1:
            res = Tensor([val_1 + val_2 for val_1, val_2 in zip(self.__data, other.__data)])

        # Matrix addition
        elif len(self.shape) == 2:
            res = Tensor([[Value(0) for _ in range(self.shape[1])] for _ in range(self.shape[0])])

            for row in range(self.shape[0]):
                for col in range(self.shape[1]):
                    res[row][col].__data = self.__data[row][col] + other.__data[row][col]
        
        return res

    def __repr__(self):
        return f"Tensor({self.__data})"
    
    def __getitem__(self, index):
        return Tensor(self.__data[index])
    
    def __len__(self):
        return len(self.__data)
    
    def transpose(self):
        
        """
            Transpose the current object. Only applicable to scalars, vectors or matrices.
        """

        if len(self.shape) > 2:
            raise Exception("Error! Transposition is only defined for scalars, vectors or matrices! (0-dimension, 1-dimension or 2-dimension tensors)")

        new_data = [list(row) for row in zip(*self.__data)]

        return Tensor(new_data)