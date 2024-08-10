"""
    minitorch.optim is a minitorch package implementing various optimization algorithms.

    To use minitorch.optim you have to construct an optimizer object that will hold the current state and will update the parameters based on the computed gradients.

    All optimizers implement a step() method, that updates the parameters.
"""
from .SGD import SGD