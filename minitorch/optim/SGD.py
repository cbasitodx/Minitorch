from minitorch.optim.Optimizer import Optimizer
from minitorch.Autograd import Value
from typing import Iterable

class SGD (Optimizer):
    """
    ===========
    **Summary**
    ===========

        Class that implements stochastic gradient descent (SGD).
    
    ==============
    **Parameters**
    ==============
    
        * **params:** (*Iterable[Value]*) An iterable of Value objects that are subject to be optimized by the optimizer algorithm (Optimizer subclass).
        * **lr:** (*float*) Learning rate to be used during the stochastic gradient descent. Default is :math:`1 \\cdot 10^{-3}`. 

    ======================
    **Instance Variables**
    ======================

        * **params:** (*Iterable[Value]*) Parameters of the model that must be optimized.
        * **lr:** (*float*) Learning rate hyperparameter for the optimization algorithm.

    ===========
    **Example**
    ===========

        .. code-block:: python

            # This is a basic training loop

            optim_sgd = SGD(model.parameters())

            for i in range(epochs):
                for output, label in zip(data):
                
                    output = model.forward(d)
                    model.zero_grad()
                    loss = MSE(output, label)
                    loss.backward()
                    optim_sgd.step()

    """
    def __init__(self, params : Iterable[Value], lr : float = 1e-3):
        super().__init__(params)
        self.lr = lr
    
    def step(self) -> None:
        """
            Performs a single optimization step.

            Optimization step for SGD is defined as:

            .. math::
                \\text{for } t = 1 \\text{ to ... do} \\\\
                \\begin{align*}
                    &g_t \\leftarrow \\nabla_{\\theta} f_t(\\theta_{t-1}) \\\\
                    &\\theta_t \\leftarrow \\theta_{t-1} - \\gamma \\cdot g_t
                \\end{align*}

            Where :math:`\\gamma` is the learning rate and :math:`t` goes from 1 to the parameters of the model.
                
        """
        # Iterate through all the parameters and update them "in the direction of their gradient with a step as big as {learning rate}"
        for param in self.params:
            param.data = param.data - (self.lr * param.grad)
    