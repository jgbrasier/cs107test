Usage
=====

.. _installation:

Installation
------------

Create a virtual environment, for example *autodiff_107-env*, but you are free to name it as you wish!

.. code-block:: console

    $ python3 -m venv /path/to/autodiff_107-env

Activate it!

.. code-block:: console

    $ source autodiff_107-env/bin/activate

pip install **autodiff_107**. WARNING!! Make sure you are using the latest version of pip!

.. code-block:: console

    $ python3 -m pip install --upgrade pip

.. code-block:: console

    $ pip install autodiff_107
   
Congrats! You are now all set to use the package.

.. _getting_started:

Getting Started
---------------

Creating a variable is the same as creating a numpy array, in fact our variables are built out of numpy arrays!

Example:

.. code-block:: python3

    import autodiff_107.diff as ad

    x = ad.variable([[1,2],[3,4]])

To see the numerical value of a variable you have to use the **value** function:

.. code-block:: console

    >>> ad.value(x)
    array([[1, 2],
           [3, 4]])

Our variables can be used just like numpy.ndarrays! Feel free to use matrix operators as well as numpy mathematical functions but be sure to use our version of numpy which supports operations on our variables.

.. code-block:: python3

    from autodiff_107.math import numpy as np

.. _forward_mode:

Forward Mode
------------

Like most of our library, forward mode is very simple to use. First we want to import the **diff** module which contains our automatic differentiation implementation and numpy:

.. code-block:: python3

    import autodiff_107.diff as ad
    from autodiff_107.math import numpy as np

Next we can set up a variable we want to use in our function (or multiple)

.. code-block:: python3

    x = ad.variable([2,4,5,9])

Now we need to set a seed for this variable. By default the seed for all Nodes is 0. We can either set the seed component individually for each input or we can set it for the entire variable (this is probably the best way in most cases).

.. code-block:: python3

    seed = np.array([0,1,0,0])
    ad.set_fm_seed(x, seed)

Now, whenever we compute a function with **x**, **f** will automatically have the derivative of itself with respect to **x** and its defined seed. For example we could have the following function:

.. code-block:: python3

    f = x @ x + 3 * x

We would expect the derivative of **f** with respect to **x** and **seed** to be 2 x 4 + 3 for the second element and 2x4 for the rest and indeed, checking **f**'s derivative we get:

.. code-block:: console

    >>> ad.get_fm_derivative(f)
    array([ 8., 11.,  8.,  8.])

Note that when you set a new seed, you have to recompute **f** as such:

.. code-block:: console

    >>> f = lambda x: x @ x + 3 * x
    >>> ad.set_fm_seed(x, np.array([0,0,1,0]))
    >>> ad.get_fm_derivative(f(x))
    array([10., 10., 13., 10.])

The result is the same but it will be easier to avoid mistakes.

Note that we place no restrictions on the seed passed into **set_fm_seed**, it does not have to sum to one and could have fractions. Usage of this kind would be pretty advanced and anyone looking to use forward mode this way should already know what they are doing.

If you want to check the current seed of a variable use:

.. code-block:: console

    >>> ad.get_fm_seed(x)
    array([0, 0, 1, 0])

Reverse Mode
------------

Reverse Mode is even easier to use than forward mode! The most important thing you have to remember is to wrap your variables in our variable function.

.. code-block:: python3

    import autodiff_107.diff as ad
    from autodiff_107.math import numpy as np

    x = ad.variable([[1,2],[3,4]])
    y = np.array([[3,9],[1,5]])

    f = x @ y + y * x

The result of this operation is

.. code-block:: console

    >>> ad.value(f)
    array([[ 8, 37],
           [16, 67]])





