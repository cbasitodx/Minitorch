from minitorch.Autograd import Value
from typing import Iterable

class Optimizer:
    """
    ===========
    **Summary**
    ===========

        Base class that all optimizers must import. Functions as an interface and provides high abstraction methods.

    ==============
    **Parameters**
    ==============
    
        * **params:** (*Iterable[Value]*) An iterable of Value objects that are subject to be optimized by the optimizer algorithm (Optimizer subclass).

    ======================
    **Instance Variables**
    ======================

        * **params:** (*Iterable[Value]*) Parameters of the model that must be optimized.

    """
    
    def __init__(self, params : Iterable[Value]):
        self.params = params

    def step(self) -> None:
        """
            Performs a single optimization step.
        """
        pass

    def zero_grad(self) -> None:
        """
            Resets the gradients of all optimized parameters.
        """
        for param in self.params:
            param.grad = 0