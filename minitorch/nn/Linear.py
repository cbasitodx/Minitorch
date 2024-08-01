from minitorch.engine.Autograd import Value
from minitorch.tensor.Tensor import Tensor
from minitorch.nn.Module import Module
from typing import List
import random
import math


class Linear(Module):

    def __init__(self, in_features : int, out_features : int, bias : bool = True):
        
        k = 1/in_features
        self.__weights = Tensor([[random.uniform(-math.sqrt(k), math.sqrt(k)) for _ in range(in_features)] for _ in range(out_features)]) # TODO: SHAPE DE ESTA MATRIZ ES (out, in)

        self.__bias : Tensor
        if bias:
            self.__bias = Tensor([random.uniform(-math.sqrt(k), math.sqrt(k)) for _ in range(out_features)])
        else:
            self.__bias = Tensor([0 for _ in range(out_features)])

    def __call__(self, activation : Tensor):
        
        # TODO: PONER COMPROBACIONES DE TAMANO AQUI!!!

        return (activation * self.__weights.transpose()) + self.__bias
    
    def parameters(self):
        
        # TODO: PONER UN METODO FLATTEN EN LA CLASE TENSOR
        params_weights : List[Value] = []
        for row in self.__weights:
            for w in row:
                params_weights.append(w)

        param_bias : List[Value] = [b for b in self.__bias]
        return params_weights + param_bias