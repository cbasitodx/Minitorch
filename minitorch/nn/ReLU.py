from minitorch.engine.Autograd import Value
from minitorch.tensor.Tensor import Tensor

from typing import List

class ReLU:

    """
    ===========
    **Summary**
    ===========

        Applies the Rectified Linear Unit (ReLU) function element-wise.

        Can be applied either to a single Value object, int or float, or to a Tensor (including all dimensions).

        ReLU function is described as:

        .. math::
            \\begin{align*}
                ReLU :=& ReLU() \\\\
                ReLu(x) =& x \\text{ if } x > 0  \\text{ } ; \\text{ } 0 \\text{ otherwise}
            \\end{align*}

        Where :math:`x` is described as a single value.

        .. note::
            This class is compatible with containers such as :class:`minitorch.nn.Sequential`

    ==============
    **Parameters**
    ==============
    
        * **res:** (:class:`minitorch.tensor.Tensor` | :class:`minitorch.engine.Value`) Result of the operation. Serves for caching.

    ===========
    **Example**
    ===========

        >>> r = ReLU()
        >>> x = Tensor([1,-2,3])
        >>> r(x) # >>> Tensor([Value(data=1, grad=0), Value(data=0, grad=0), Value(data=3, grad=0)])

    """    

    def __init__(self):
        self.res : Tensor | Value = None
    
    def __call__(self, activation : Tensor | Value | int | float) -> Tensor | Value:
        
        # If the input is a Tensor
        if isinstance(activation, Tensor):
            
            # This auxiliar function takes the activation Tensor (as a list), and iterates through it.
            # It builds a list of Value objects where each element is the ReLU evaluated at the element in the original Tensor.
            def calculate_relu_element_wise(act : List) -> List[Value] | List[List[Value]] | List[List[List[Value]]]:
                if isinstance(act, list):
                    return [calculate_relu_element_wise(item) for item in act]
                elif isinstance(act, Value):
                    return act.relu()
                
                # NOTE: No error handling here because Tensor class assures integrity
                #       over the structure we are iterating through
            
            self.res = Tensor(calculate_relu_element_wise(list(activation)))
        
        # If the input is a single Value object
        elif isinstance(activation, Value):
            self.res = activation.relu()
        
        # If the input is an integer or a float
        elif isinstance(activation, (int,float)):
            wrapped : Value = Value(activation)
            self.res = wrapped.relu
        
        else:
            raise Exception("ReLU can ONLY be applied to Tensor objects, Value objects, integers or floats")

        return self.res