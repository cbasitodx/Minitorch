import math
from typing import Tuple

from graphviz import Digraph

class Value:

    """
    ===========
    **Summary**
    ===========

        Class that represents a real value variable in a function. 
        Objects of this class interact with themselves in a tree fashion when within the same function.
        This structure allows the numerical calculation of partial derivatives of the function with respect to every Value object that comprises it.
        It even allows numerical calculation of partial derivatives of sub functions. This is accomplished via **reverse automatic differentiation**.

        Objects of this class support **addition**, **substraction**, **multiplication** and **exponentiation** between them. They can also be **printed**.

        **ReLU** and **Sigmoid** functions of an object of this class can also be calculated.  
    
    ==============
    **Parameters**
    ==============
    
        * **data** (*float*) Numerical value of the object.

        * **grad:** (*float*) Numerical value of the partial derivative of the function with respect to the current object evaluated in an initial value.

        * **label:** (*str*) Symbolic name of the variable.

        * **__backward:** (*callable*) Function that explicits how the function that this object is part of must be differentiated with respect to this object.

        * **__children:** (*Set*) Tuple of objects of this class that are combined together with an operation to form the current object.

        * **__op:** (*str*) Symbolic representation of the operation applied to __children in order to form the current object.

    ======================
    **Instance Variables**
    ======================

        * **data:** (*float*) Numerical value of the object.
        
        * **children:** (*tuple*) Tuple of objects of this class that are combined with an operation to form the current object.
        
        * **op:** (*str*) Operation that forms this object.
        
        * **label:** (*str*) Label of this object.

    ===========
    **Example**
    ===========

        >>> a = Value(3)
        >>> b = Value(4)
        >>> f = a * b + (b**2)

    """


    def __init__(self, data : float, children=(), op='', label="") -> None:
        
        # **** Public attributes **** #

        self.data : float = data
        
        self.grad : float = 0
        
        self.label : str = label
        
        # **** Private attributes **** #
        
        self.__backward : callable = lambda: None  
        
        self.__children : Tuple['Value'] = tuple(children) 
        
        self.__op : str = op


    def __wrapValue(self, other : int | float) -> 'Value':

        """
            Wraps a numerical value (integer or float) as an object of this class.
        """

        # We check if other is a 'Value' class instance
        if not isinstance(other, Value):

            # Check if its a numeric type
            if isinstance(other, (int, float)) and not isinstance(other, bool):
                # Wrap into a value class
                return Value(float(other), label=str(other))
            
            # other can only by 'Value' or numeric type
            else:
                raise Exception("Type error: trying to operate a non numeric value!")
        
        else: 
            return other

    def __add__(self, other : 'Value') -> 'Value':
        
        other = self.__wrapValue(other)
        
        out =  Value(self.data + other.data, children=(self, other), op = "+", label=f"{self.label}+{other.label}")

        def __backward():
            self.grad  += 1*out.grad
            other.grad += 1*out.grad
        
        out.__backward = __backward 

        return out

    def __mul__(self, other : 'Value') -> 'Value':

        other = self.__wrapValue(other)
        
        out =  Value(self.data*other.data, children=(self, other), op = "*", label=f"{self.label}*{other.label}")

        def __backward():
            self.grad  += other.data*out.grad
            other.grad += self.data*out.grad
        out.__backward = __backward 

        return out
    
    def __pow__(self, other : 'Value') -> 'Value':
        
        other = self.__wrapValue(other)
        
        out =  Value(self.data**(other.data), children=(self, other), op = "^", label=f"{self.label}^{other.label}")

        def __backward():
            self.grad  += other.data*(self.data**(other.data - 1))*out.grad
            
            #NOTE: Commented because line above gives problems with logarithms (e.g: negative arguments)
            #other.grad += (self.data**(other.data))*math.log(self.data)*out.grad
        out.__backward = __backward 

        return out        

    def sigmoid(self) -> 'Value':

        """
            Evaluates the sigmoid function (see below) with this object as argument.
            
            .. math::

                sig(x) = \\sigma(x) = \\frac{1}{ 1+e^{-x} }

        """

        s = 1/(1+math.exp(-self.data))
        out =  Value(s, children=(self,), op = "sig", label=f"sig({self.label})")

        def __backward():
            self.grad  += (s*(1-s))*out.grad
        out.__backward = __backward 

        return out       

    def relu(self) -> 'Value':

        """
            Evaluates the ReLU (Rectifie Linear Unit) function (see below) with this object as argument.
            
            .. math::

                ReLU(x) = x \\text{ if } x > 0  \\text{ } ; \\text{ } 0 \\text{ otherwise}

        """

        r = self.data if self.data > 0 else 0
        out =  Value(r, children=(self,), op = "ReLU", label=f"ReLU({self.label})")

        def __backward():
            self.grad  += 1*out.grad if self.data > 0 else 0
        out.__backward = __backward 

        return out

    def log(self) -> 'Value':

        """
            Compute the natural logarithm (logarithm base *e*) with this object as argument.
        """

        # Error handling
        if self.data <= 0:
            raise Exception("Errro! Natural logarithm is not defined in the real numbers for values lesser than or equal to zero")

        out = Value(math.log(self.data), children=(self,), op="log", label=f"log({self.label})")

        def __backward():
            self.grad += (1/self.data)*out.grad
        out.__backward = __backward
        
        return out

    def __truediv__(self, other : 'Value') -> 'Value':
        return self*(other**(-1))

    def __neg__(self) -> 'Value':
        return self * (-1)

    def __sub__(self, other : 'Value') -> 'Value':
        return self + (-other)
    
    def __radd__(self, other : 'Value') -> 'Value':
        return self + other

    def __rsub__(self, other : 'Value') -> 'Value':
        return other + (-self)
    
    def __rmul__(self, other : 'Value') -> 'Value':
        return self*other
    
    def __rtruediv__(self, other : 'Value') -> 'Value':
        return (self**(-1))*other

    def __rpow__(self, other) -> 'Value':
        other = self._wrapValue(other)
        return other**self

    def __repr__(self) -> str:
        return f"Value(data={self.data}, grad={self.grad})"

    def backward(self):

        """
            Differentiates the **function whose value is the current object** with respect to all of the nodes in the tree whose root is the current object.
            Differentiation is performed using **reverse automatic differentiation**.
        """
        
        # Topological sort of the tree is performed in order to iterate non-recursively through all the nodes.
        # Topological sort is prefered for this task for the order of the returned list
        topo = []
        visited = set()

        def build_topo(v : Value) -> None:
            # This method fills the 'visited' list using topological sort. Takes as root the argument 'v'
            if v not in visited:
                visited.add(v)
                for child in v.__children:
                    build_topo(child)
                topo.append(v)

        build_topo(self)
        
        # We set the gradient of the root node to 1 (partial derivate of the function w.r.t itself is 1)
        self.grad = 1
        
        # Differentiate the function w.r.t all the other Value nodes,
        for i in range(len(topo) - 1, 0, -1):
            v = topo[i]
            v.__backward()

    # Useful methods for visualization of the computation graph
    def getChildren(self) -> set:

        """
            Return the children of the current object in a **set**.
        """

        return self.__children
    
    def getOperation(self) -> str:
        
        """
            Return the operation of the current object as a **string**.
        """

        return self.__op
    
    def visualize(self) -> Digraph:

        """
            This method draws the tree structure that comprises the function to be differentiated.
            The root node of the tree is the object who makes the call to the method.

            .. note::
                All credits for the source code of this method go to Andrej Karpathy.
                Please refer to his `video <https://www.youtube.com/watch?v=VMj-3S1tku0>`_
            
            .. note::
                For more information about return type, please refer to `graphviz official documentation <https://graphviz.readthedocs.io/en/stable/index.html>`_
        """

        def trace(root):
            # builds a set of all nodes and edges in a graph
            nodes, edges = set(), set()
            
            def build(v):
                if v not in nodes:
                    nodes.add(v)
                    
                    for child in v.getChildren():
                        edges.add((child, v))
                        build(child)
            build(root)
            return nodes, edges

        def draw_dot(root):
            dot = Digraph(format='svg', graph_attr={'rankdir': 'LR'}) # LR = left to right

            nodes, edges = trace(root)

            for n in nodes:
                uid = str(id(n))
                # for any value in the graph, create a rectangular ('record') node for it
                dot.node(name = uid, label = "{ %s | data %.4f | grad %.4f }" % (n.label, n.data, n.grad), shape='record')

                if n.getOperation():
                    # if this value is a result of some operation, create an op node for it
                    dot.node(name = uid + n.getOperation(), label = n.getOperation())
                    # and connect this node to it
                    dot.edge(uid + n.getOperation(), uid)

            for n1, n2 in edges:
                # connect n1 to the op node of n2
                dot.edge(str(id(n1)), str(id(n2)) + n2.getOperation())

            return dot
        
        return draw_dot(self)
