from __future__ import annotations
from minitorch.Autograd import Value
from typing import List, Tuple, Any
from copy import deepcopy

class Tensor:

    """
    ===========
    **Summary**
    ===========

        A :class:`Tensor` is a multi-dimensional arrange of numbers containing real-valued elements.

        This class only allows a maximum of four (3) dimensions. These can be seen as:
            
        * **1 dimensional tensor**:
            Is a **vector**. It's initialized with a list. **Scalar values can be initialized with one (1) element lists**

        * **2 dimensional tensor**:
            Is a **matrix**. It's initialized with a list of lists.
        
        * **3 dimensional tensor** & **4 dimensional tensor**:
            Is called a **multi-index object** in mathematics (**tensor** word is reserved for multilinear applications in mathematics). It's initialized with a 3-dimensional or 4-dimensional list.
            The 3-dimensional tensor is useful for images that contains various channels (e.g a color image), while the 4-dimensional tensor is useful for representing **batches of data**, being the fourth dimension the batch size.
         
        .. warning::

            Uniformity along all dimensions is assumed. This means that, for example, the matrix [[1,2], [3]] is **not** supported as a tensor.
            Not mantaining uniformity may lead to undefined behaviour. 
            
    ==============
    **Parameters**
    ==============
    
        * **__shape:** (*Tuple[int]*) Dimensions of the tensor. Its read as (Batch Size, Number of Channels, Rows, Columns)

        * **__data:** (:class:`Autograd.Value` | *List*) List that contains the raw data of the tensor as Value objects.

        * **__dim:** (*int*) Length of self.__data. Is used for convenience inside the class

    ======================
    **Instance Variables**
    ======================

        * **data:** (*int | float | Value | List | List[List] | List[List[List] | List[List[List[List]]]*) List containing the raw data of the tensor.

    ===========
    **Example**
    ===========

        >>> three_dim_tens = Tensor([ [[1,2], [3,4], [4,5]], [[5,6], [7,8]] ])
        >>> scalar = Tensor([3.14])

    """
        
    def __init__(self, 
                 data : int | float | Value | List | List[List] | List[List[List]] | List[List[List[List]]]):

        # Empty lists are not allowed as instance variables
        if isinstance(data, list) and len(data) == 0:
            raise Exception("Error! Empty lists are invalid instance variables for this class")

        self.__shape : Tuple[int]
        
        def get_dims(data : List):
            if type(data) != list:
                return []
            else:
                return [len(data)] + get_dims(data[0]) 
        
        self.__shape = tuple(get_dims(data))

        self.__dim = len(self.__shape)
        
        if self.__dim > 4 or self.__dim < 1:
            raise Exception("Error! Tensors with more than four (4) dimensions or less than one (1) are not supported")

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
                raise Exception("Error! Unable to wrap the raw data values inside Value objects. Elements of the tensor MUST be real-valued numbers")
        
        self.__data = get_value_list(data)

        
    def __iter__(self):
        return iter(self.__data)
    
    def __scalar_vect_mul(self, other : int | float | Value) -> Tensor:
        """
            Perform Scalar - Vector multiplication
        """
        return Tensor([other * elem for elem in self.__data])
        
    def __scalar_matrix_mul(self, other : int | float | Value) -> Tensor:
        """
            Perform Scalar - Matrix multiplication
        """
        new_col : List = []
        res_aux : List = []
        for row in self.__data:
            res_aux.append([other * col for col in row])

        return Tensor(res_aux)
    
    def __dot_product(self, other : Tensor) -> Tensor:
        """
            Perform the dot product between this object and other Tensor object. Both being 1-dimensional Tensor objects
        """
        # Firstly, check length matching
        if len(self) != len(other):
            raise Exception("Error! Dot product can only be performed between vectors (1-dimensional Tensor objects) of the same lenghth")
        
        else:
            return Tensor([sum(self.__data[i] * other.__data[i] for i in range(len(self.__data))), ])

    def __vector_matrix_mul(self, other : Tensor) -> Tensor:
        """
            Perform Vector - Matrix multiplication. Vector must be a "row vector"
        """
        # Number of elements in the resulting vector (1-dimensional Tensor object). It has as many elements as columns the 'other' object has
        elems : int = other.__shape[-1]

        # Number of rows and columns in the 'other' object
        cols_other : int = other.__shape[-1]
        rows_other : int = other.__shape[-2]

        res_aux : List = [0]*elems

        for i in range(cols_other):
            res_aux[i] = sum([self.__data[j] * other.__data[j][i] for j in range(rows_other)])

        return Tensor(res_aux)

    def __matrix_matrix_mul(self, other: Tensor) -> Tensor:
        """
            Perform Matrix - Matrix multiplication in a strict mathematical sense
        """
        # Extract rows and columns of the current object
        self_rows : int = self.__shape[-2]
        self_cols : int = self.__shape[-1]

        # Extract rows and columns of the 'other' object
        other_rows : int = other.__shape[-2]
        other_cols : int = other.__shape[-1]

        res_aux : Tensor = Tensor([[Value(0) for _ in range(other_cols)]for _ in range(self_rows)])

        for self_row in range(self_rows):
            for other_col in range(other_cols):
                for other_row in range(other_rows):
                    res_aux.__data[self_row][other_col] += self.__data[self_row][other_row] * other.__data[other_row][other_col]

        # NOTE: This is done for dropping dimensions (flattening), e.g: [[1]] -> [1]
        while res_aux.__dim != 1 and res_aux.shape()[0] == 1:
            res_aux = res_aux[0]

        return res_aux

    def __mul__(self, other : int | float | Value | Tensor) -> Tensor:

        """
            In version 0.0.1, this class only supports mathematically accurate operations. These are:

            * Scalar - Vector multiplication
            * Scalar - Matrix multiplication
            * Vector dot product
            * Matrix - Vector multiplication (and viceversa)
            * Matrix - Matrix multiplication
        """

        res : Tensor

        # Scalar - Vector multiplication
        if self.__dim == 1 and isinstance(other, (int, float, Value)):
            res = self.__scalar_vect_mul(other)

        # Scalar - Matrix multiplication
        elif self.__dim == 2 and isinstance(other, (int, float, Value)):
            res = self.__scalar_matrix_mul(other)
        
        
        # Vector dot product
        elif self.__dim == 1 and isinstance(other, Tensor) and other.__dim == 1:
            res = self.__dot_product(other)
        
        # Vector - Matrix multiplication
        elif self.__dim == 1 and isinstance(other, Tensor) and other.__dim == 2:

            # Check if self.col == other.row
            if self.__shape[-1] != other.__shape[-2]:
                raise Exception("Error! Dimensions mismatch, unable to perform multiplication!")
            else:
                res = self.__vector_matrix_mul(other)
        
        # Matrix - Matrix multiplication (Matrix - Vector multiplication enters in this category because, in that case, the vector needs to be a "column vector", wich is just a matrix in terms of this object)
        elif self.__dim == 2 and isinstance(other, Tensor) and other.__dim == 2:
            
            # Check if self.col == other.row
            if self.__shape[-1] != other.__shape[-2]:
                raise Exception("Error! Dimensions mismatch, unable to perform multiplication!")
            else:
                res = self.__matrix_matrix_mul(other)

        else:
            raise Exception("Error! Multiplication is not supported dimensions grater than 3")

        return res
    
    def __vect_add(self, other : Tensor) -> Tensor:
        return Tensor([val_1 + val_2 for val_1, val_2 in zip(self.__data, other.__data)])
    
    def __matrix_add(self, other : Tensor) -> Tensor:
        res_aux : Tensor = Tensor([[Value(0) for _ in range(self.__shape[1])] for _ in range(self.__shape[0])])

        for row in range(self.__shape[0]):
            for col in range(self.__shape[1]):
                res_aux.__data[row][col] = self.__data[row][col] + other.__data[row][col]
        
        return res_aux


    def __add__(self, other : Tensor) -> Tensor:
        
        # Check if the tensors have the same dimentions and that they are, at most, 2-dimensional
        if self.__shape != other.__shape or self.__dim > 2 or other.__dim > 2:
            raise Exception("Error! Addition is only defined between tensors of the same dimention, and is not supported (as for v.1.0.0) for tensors that have more than 2 dimensions")

        res : Tensor

        # Vector addition
        if self.__dim == 1:
            res = self.__vect_add(other)

        # Matrix addition
        elif self.__dim == 2:
            res = self.__matrix_add(other)
        
        return res
    
    def __sub__(self, other : Tensor) -> Tensor:
        # Check if the tensors have the same dimentions and that they are, at most, 2-dimensional
        if self.__shape != other.__shape or self.__dim > 2 or other.__dim > 2:
            raise Exception("Error! Substraction is only defined between tensors of the same dimention, and is not supported (as for v.1.0.0) for tensors that have more than 2 dimensions")

        res : Tensor

        # Vector substraction
        if self.__dim == 1:
            res = self.__vect_add(other * (-1))

        # Matrix substraction
        elif self.__dim == 2:
            res = self.__matrix_add(other * (-1))
        
        return res

    def __repr__(self):
        return f"Tensor({self.__data})"
    
    def __getitem__(self, index):
        if self.__dim == 1:
            return Tensor([self.__data[index],]) 
        else:
            return Tensor(self.__data[index])
    
    def __len__(self):
        return len(self.__data)
    
    def to_list(self) -> List:
        """
            Return the Tensor object as a python list
        """
        return deepcopy(self.__data)

    def item(self) -> Value:
        """
            Only usable for 1-dimensional Tensor objects that have one element: returns a the element of the Tensor object wrapped in a Value object.
        """
        if self.__dim == 1 and len(self.__data) == 1:
            return self.__data[0]
        else:
            raise Exception("Error! This method is not applicable for Tensor objects that have more than one element or that have a dimension greater than one (1). In this case, refer to the method 'to_list'")

    def shape(self) -> Tuple[int]:
        """
            Returns the shape of this object. Shape is interpreted as (Batch Size, Number of Channels, Rows, Columns)
        """
        return deepcopy(self.__shape)

    def transpose(self):
        """
            Transpose the current object. Only applicable to scalars, vectors or matrices.
        """

        if self.__dim > 2:
            raise Exception("Error! Transposition is only defined for scalars, vectors or matrices! (0-dimension, 1-dimension or 2-dimension tensors)")

        new_data = [list(row) for row in zip(*self.__data)]

        return Tensor(new_data)