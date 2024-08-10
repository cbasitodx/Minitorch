from typing import List
from minitorch.Autograd import Value

class Module:

    """
    ===========
    **Summary**
    ===========

        Base class for all neural network modules.
        
        Your models should also subclass this class.

        Allows basic functionalities, like resetting the gradient ("zeroing out") of the parameters of each layer and getting said parameters.
        This last functionality is intended for custom models classes. Classes relative to the package overwrite this method.
    """

    def zero_grad(self) -> None:

        """
            Resets (sets to 0) the gradient of every parameter in the model.
            This is done in order to avoid accumulation errors.
        """

        for p in self.parameters():
            p.grad = 0

    def parameters(self) -> List[Value]:

        """
            Returns the parameters of the model as a list of Value objects.
            
            .. warning::
                This method should **NOT** be overwriten by custom model classes.
        """

        # List of parameters (Value objects) to be returned
        params : List[Value] = []
        
        # Iterate through all the attributes of the object,
        # if the attribute is a Module subclass, the call the
        # parameters() method and concatenate it to the result list
        for attr in self.__dict__.values():
            if isinstance(attr, Module):
                params += attr.parameters()

        return params
                
