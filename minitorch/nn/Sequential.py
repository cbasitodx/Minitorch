from minitorch.engine.Autograd import Value
from minitorch.tensor.Tensor import Tensor
from minitorch.nn.Module import Module
from typing import Tuple, List


class Sequential(Module):

    """
    ===========
    **Summary**
    ===========

        A sequential container.

        Modules will be added to it in the order they are passed in the constructor.

        The value a Sequential provides over manually calling a sequence of modules is that it allows treating the whole container as a single module.

       Layers in a Sequential are connected in a cascading way.

    ==============
    **Parameters**
    ==============
    
        * **__layers:** (*Tuple[callable]*) Tuple of all the callable objects passed to the Sequential.
        * **__trainable:** (*List[callable]*) List of all the callable objects passed to the Sequential that are *trainable* (this meaning, that subclass :class:`minitorch.nn.Module`).

    ======================
    **Instance Variables**
    ======================

        * ***layers:** (*Tuple[callable]*) Tuple of all the callable objects that comprise a Sequential instance.

    ===========
    **Example**
    ===========

        .. code-block:: python
        
            class MLP(Module):
                def __init__(self):
                    self.layers = Sequential(
                        Linear(2,3),
                        ReLU(),
                        Linear(3,1),
                        Sigmoid()
                    )
                
                def forward(self, input : List[float]):
                    
                    # We convert the input to a tensor
                    input = Tensor(input)
            
                    # Now we do the forward pass
                    return self.layers(input)

    """

    def __init__(self, *layers):

        self.__layers : Tuple[callable] = layers

        self.__trainable : List[callable] = []
        for layer in self.__layers:
            if isinstance(layer, Module):
                self.__trainable.append(layer)
        
    def __call__(self, input_vect : Tensor):

        activation = (self.__layers[0])(input_vect)
        for i in range(1, len(self.__layers)):
            activation = (self.__layers[i])(activation)
        return activation

    def parameters(self):
        parameters : List[Value] = []
        for layer in self.__trainable:
            parameters += layer.parameters()

        return parameters 