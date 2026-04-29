"""KernelBench adapter — unimplemented placeholder namespace.

No ``load`` callable is exported yet; importing this module is a no-op.
A future phase will wrap ``Model.forward`` into a ``def run(*inputs)``
string in ``Definition.reference``, with init params handled via
``custom_inputs_entrypoint``. Until then the ``optimize.py`` dispatcher
raises ``NotImplementedError`` for the ``kernelbench`` adapter route.
"""
