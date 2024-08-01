from minitorch.engine.Autograd import Value
from minitorch.tensor.Tensor import Tensor


class ReLU:
    def __init__(self):
        self.res : Tensor | Value = None
    
    def __call__(self, activation : Tensor | Value):

        if isinstance(activation, Tensor):
            self.res = Tensor([neuron_act.relu() for neuron_act in activation])
        
        elif isinstance(activation, Value):
            self.res = activation.relu()
        
        return self.res