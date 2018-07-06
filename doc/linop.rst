Linear Operators
================

.. _linop:

linop interface
--------------------

The linop interface provides an abstraction for linear operators.
For each linop :math:`A`, it supports forward operation :math:`x \rightarrow A(x)`, adjoint operation :math:`x \rightarrow A^\top(x)` and normal operation :math:`x \rightarrow A^\top A(x)`.

.. automodule:: sigpy.linop
