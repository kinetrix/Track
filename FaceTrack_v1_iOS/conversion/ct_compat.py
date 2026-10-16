"""coremltools 9.0 + NumPy 2.x workaround.

coremltools' torch `int`/`bool` cast does `int(x.val)` on a folded constant; with
NumPy >= 2 that raises "only 0-dimensional arrays can be converted to Python
scalars" when the constant is a 1-element array (happens for shape arithmetic in
timm/SAM 2 code). Patch it to use `.item()`. Import this before converting.
"""
import numpy as np
from coremltools.converters.mil import Builder as mb
from coremltools.converters.mil.frontend.torch import ops as _ops


def _cast(context, node, dtype, dtype_name):
    x = _ops._get_inputs(context, node, expected=1)[0]
    if not (len(x.shape) == 0 or np.all([d == 1 for d in x.shape])):
        raise ValueError("input to cast must be either a scalar or a length 1 tensor")
    if x.can_be_folded_to_const():
        val = x.val
        res = x if isinstance(val, dtype) else mb.const(val=dtype(np.asarray(val).item()), name=node.name)
    elif len(x.shape) > 0:
        res = mb.cast(x=mb.squeeze(x=x, name=node.name + "_item"), dtype=dtype_name, name=node.name)
    else:
        res = mb.cast(x=x, dtype=dtype_name, name=node.name)
    context.add(res, node.name)


_ops._cast = _cast
