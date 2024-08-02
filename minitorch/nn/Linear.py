from minitorch.engine.Autograd import Value
from minitorch.tensor.Tensor import Tensor
from minitorch.nn.Module import Module
from typing import List
import random
import math


class Linear(Module):

    """
    ===========
    **Summary**
    ===========

        Applies a linear transformation to the incoming data

        .. math::
            \\begin{align*}
                l :=& Linear \\\\
                l(\\bar{x}) =& \\bar{x} \\cdot W^{T} + \\bar{b}
            \\end{align*}

        Where the data :math:`\\bar{x}` is a **row** vector. This meaning that its shape is :math:`(1,in_features)`,
        :math:`W` is a weight matrix of shape :math:`(out_features, in_features)` and :math:`\\bar{b}` is the bias vector of shape :math:`(1, out_features)`.

        Both :math:`W` and :math:`\\bar{b}` are **learnable parameters**. The bias :math:`\\bar{b}` is **optional** (if ``bias = False``, then it is initialized as a zero vector).

        Elements of the weight matrix :math:`W` are sampled initially from :math:`\\mathcal{U}(-\\sqrt{\\frac{1}{in_features}}, \\sqrt{\\frac{1}{in_features}})` 

    ==============
    **Parameters**
    ==============
    
        * **__weights:** (:class:`minitorch.tensor.Tensor`) Learnable weights. Matrix of shape :math:`(out_features, in_features)`.
        * **__bias:** (:class:`minitorch.tensor.Tensor`) Learnable bias vector. If ``bias = True``, its a vector of shape :math:`(1, out_features)`.

    ======================
    **Instance Variables**
    ======================

        * **in_features:** (*int*) Number of input features. 
        * **out_features:** (*int*) Number of output features. 

    ===========
    **Example**
    ===========

        >>> l = Linear(2,3)
        >>> x = Tensor([5, 6])
        >>> l(x) # This returns a tensor of shape [1,3]!

    """

    def __init__(self, in_features : int, out_features : int, bias : bool = True):
        
        k : float = 1/in_features
        self.__weights : Tensor = Tensor([[random.uniform(-math.sqrt(k), math.sqrt(k)) for _ in range(in_features)] for _ in range(out_features)])

        self.__bias : Tensor
        if bias:
            self.__bias = Tensor([random.uniform(-math.sqrt(k), math.sqrt(k)) for _ in range(out_features)])
        else:
            self.__bias = Tensor([0 for _ in range(out_features)])

    def __call__(self, activation : Tensor):
        
        # TODO: PONER COMPROBACIONES DE TAMANO AQUI!!!

        return (activation * self.__weights.transpose()) + self.__bias
    
    def parameters(self) -> List[Value]:

        """
            Returns parameters of the model (weights and bias) as a list of Values!
        """
        
        # TODO: PONER UN METODO FLATTEN EN LA CLASE TENSOR
        params_weights : List[Value] = []
        for row in self.__weights:
            for w in row:
                params_weights.append(w)

        param_bias : List[Value] = [b for b in self.__bias]
        return params_weights + param_bias