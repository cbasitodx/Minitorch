from minitorch.engine.Autograd import Value
from minitorch.tensor.Tensor import Tensor
from minitorch.nn.Module import Module
from typing import Tuple, List


class Sequential(Module):

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