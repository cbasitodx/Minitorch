from Optimizer import Optimizer

class SGD (Optimizer):
    """
    ===========
    **Summary**
    ===========

        Class that represents a real value variable in a function. 
        Objects of this class interact with themselves in a tree fashion when within the same function.
        This structure allows the numerical calculation of partial derivatives of the function with respect to every Value object that comprises it.
        It even allows numerical calculation of partial derivatives of sub functions. This is accomplished via **reverse automatic differentiation**.

        Objects of this class support **addition**, **substraction**, **multiplication** and **exponentiation** between them. They can also be **printed**.

        **ReLU** and **Sigmoid** functions of an object of this class can also be calculated.  
    
    ==============
    **Parameters**
    ==============
    
        * **data** (*float*) Numerical value of the object.

        * **grad:** (*float*) Numerical value of the partial derivative of the function with respect to the current object evaluated in an initial value.

        * **label:** (*str*) Symbolic name of the variable.

        * **__backward:** (*callable*) Function that explicits how the function that this object is part of must be differentiated with respect to this object.

        * **__children:** (*Set*) Tuple of objects of this class that are combined together with an operation to form the current object.

        * **__op:** (*str*) Symbolic representation of the operation applied to __children in order to form the current object.

    ======================
    **Instance Variables**
    ======================

        * **data:** (*float*) Numerical value of the object.
        
        * **children:** (*tuple*) Tuple of objects of this class that are combined with an operation to form the current object.
        
        * **op:** (*str*) Operation that forms this object.
        
        * **label:** (*str*) Label of this object.

    ===========
    **Example**
    ===========

        >>> a = Value(3)
        >>> b = Value(4)
        >>> f = a * b + (b**2)

    """
    def __init__(self):
        self.x = 3
    