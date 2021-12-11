Implementation
==============

.. note::
    Here, we explain what happens under the hood. Not everything discussed here is user facing!

The core data structure for our library is a **Node**, defined in **diff.py**. **Node** s make up the computational graph that we  traverse for forward mode AD as well as for forward and backward passes of reverse mode. They each have a value, a derivative with respect to each parent and a forward mode derivative used to do forward mode. This forward mode derivative is the current derivative of the Node with respect to the inputs and a user defined seed.
Nodes have overloaded operators (addition, substraction, cos, etc) such that when they are used in a computation, new nodes are created for the result of the operation, and the resulting relative derivatives are stored for each intermediate node. We also store the name of the operation that was performed (with result._operation). The result of a function is returned as one node that is the descendant of all of the inputs used in that computation.

We store partial derivatives as a dictionary because we need to store the derivative with respect to each parent. This gives us the connections of the computational graph using the keys to this dictionary and is how we do traversal for the backward pass. This is also what we use in the visualization tools.

As mentionned, we had to implement new versions of operators like cos, exp, etc. that operate on Nodes. They also handle setting derivatives and keeping track of the names of the operation. We did this by writing a wrapper for unary numpy functions. Then we defined derivatives for all the functions we wanted to support and used them to wrap the numpy functions such that they would handle Node objects and properly maintain the computational graph.

We have a _backward function in diff.py that handles doing the reverse pass on results, and calculating the derivatives of the result with respect to the inputs. We allow passing either a single Node or a vector of Nodes into _derivative. This lets us easily handle functions from :math:`\mathbb{R}^n \rightarrow \mathbb{R}^m`. Our user facing derivative function takes f and X and will output either the derivative of f with respect to X or a jacobian, depending on the shapes of f and X. f and X do have to be flat but the user can flatten their matrices however they like and reshape the output however works best for their use case.