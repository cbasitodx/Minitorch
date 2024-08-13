from minitorch import *

class BinaryCrossEntropyLoss:
    """
    ===========
    **Summary**
    ===========

        Creates a criterion that measures the Binary Cross Entropy between the target and the input probabilities:

        It is computed as:

        .. math::
            BCE(y, \\hat{y}) = -[y \\cdot log(\\hat{y}) + (1-y) \\cdot log(1-\\hat{y})]
        
        Where :math:`y` is the true label and :math:`\\hat{y}` is the prediction of the model.

    ==============
    **Parameters**
    ==============

        * **__out:** (:class:`Autograd`) Output of the loss function.

    ======================
    **Instance Variables**
    ======================

    ===========
    **Example**
    ===========

        >>> loss = BinaryCrossEntropyLoss()
        >>> pred = Tensor([0.5,])
        >>> label = Tensor([1,])
        >>> loss(pred, label)

    """
    def __init__(self):
        self.__out : Value = None

    def __call__(self, pred : Tensor, label : Tensor) -> Value:
        # Dimensionality check
        pred_dims = len(pred.shape())
        label_dims = len(label.shape())

        pred_list = pred.to_list()
        label_list = label.to_list()

        # "...if the output and label tensor are not 1-dimensional OR they are not single value tensors..."
        if (pred_dims != 1 or label_dims != 1) or (len(pred_list) != 1 or len(label_list) != 1):
            raise Exception("Error! Binary Cross Entropy Loss can only be calculated (as for this Minitorch version) for one single predicted value and a single label")
        
        # Firstly, we extract the raw Value object
        pred_val = pred.item()
        label_val = label.item()

        # Now, we calculate the BCE loss function
        self.__out = -(label_val * pred_val.log() + (1 - label_val)*((1-pred_val).log()))

        return self.__out

    def backward(self) -> None:
        """
            Computes the gradient of the Binary Cross Entropy Loss function.
        """

        # Error handling
        if self.__out == None:
            raise Exception("Error! Please compute the function first")
    
        self.__out.backward()