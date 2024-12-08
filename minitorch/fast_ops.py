from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from numba import njit, prange

from .tensor_data import (
    MAX_DIMS,
    broadcast_index,
    index_to_position,
    shape_broadcast,
    to_index,
)
from .tensor_ops import MapProto, TensorOps

if TYPE_CHECKING:
    from typing import Any, Callable, Optional

    from .tensor import Tensor
    from .tensor_data import Index, Shape, Storage, Strides

# TIP: Use `NUMBA_DISABLE_JIT=1 pytest tests/ -m task3_1` to run these tests without JIT.

# This code will JIT compile fast versions your tensor_data functions.
# If you get an error, read the docs for NUMBA as to what is allowed
# in these functions.
to_index = njit(inline="always")(to_index)
index_to_position = njit(inline="always")(index_to_position)
broadcast_index = njit(inline="always")(broadcast_index)


class FastOps(TensorOps):
    @staticmethod
    def map(fn: Callable[[float], float]) -> MapProto:
        """See `tensor_ops.py`"""
        # This line JIT compiles your tensor_map
        f = tensor_map(njit()(fn))

        def ret(a: Tensor, out: Optional[Tensor] = None) -> Tensor:
            if out is None:
                out = a.zeros(a.shape)
            f(*out.tuple(), *a.tuple())
            return out

        return ret

    @staticmethod
    def zip(fn: Callable[[float, float], float]) -> Callable[[Tensor, Tensor], Tensor]:
        """See `tensor_ops.py`"""
        f = tensor_zip(njit()(fn))

        def ret(a: Tensor, b: Tensor) -> Tensor:
            c_shape = shape_broadcast(a.shape, b.shape)
            out = a.zeros(c_shape)
            f(*out.tuple(), *a.tuple(), *b.tuple())
            return out

        return ret

    @staticmethod
    def reduce(
        fn: Callable[[float, float], float], start: float = 0.0
    ) -> Callable[[Tensor, int], Tensor]:
        """See `tensor_ops.py`"""
        f = tensor_reduce(njit()(fn))

        def ret(a: Tensor, dim: int) -> Tensor:
            out_shape = list(a.shape)
            out_shape[dim] = 1

            # Other values when not sum.
            out = a.zeros(tuple(out_shape))
            out._tensor._storage[:] = start

            f(*out.tuple(), *a.tuple(), dim)
            return out

        return ret

    @staticmethod
    def matrix_multiply(a: Tensor, b: Tensor) -> Tensor:
        """Batched tensor matrix multiply ::

            for n:
              for i:
                for j:
                  for k:
                    out[n, i, j] += a[n, i, k] * b[n, k, j]

        Where n indicates an optional broadcasted batched dimension.

        Should work for tensor shapes of 3 dims ::

            assert a.shape[-1] == b.shape[-2]

        Args:
        ----
            a : tensor data a
            b : tensor data b

        Returns:
        -------
            New tensor data

        """
        # Make these always be a 3 dimensional multiply
        both_2d = 0
        if len(a.shape) == 2:
            a = a.contiguous().view(1, a.shape[0], a.shape[1])
            both_2d += 1
        if len(b.shape) == 2:
            b = b.contiguous().view(1, b.shape[0], b.shape[1])
            both_2d += 1
        both_2d = both_2d == 2

        ls = list(shape_broadcast(a.shape[:-2], b.shape[:-2]))
        ls.append(a.shape[-2])
        ls.append(b.shape[-1])
        assert a.shape[-1] == b.shape[-2]
        out = a.zeros(tuple(ls))

        tensor_matrix_multiply(*out.tuple(), *a.tuple(), *b.tuple())

        # Undo 3d if we added it.
        if both_2d:
            out = out.view(out.shape[1], out.shape[2])
        return out


# Implementations


def tensor_map(fn: Callable[[float], float]) -> Any:
    """NUMBA low_level tensor_map function. See `tensor_ops.py` for description.

    Optimizations:

    * Main loop in parallel
    * All indices use numpy buffers
    * When `out` and `in` are stride-aligned, avoid indexing

    Args:
    ----
        fn: function mappings floats-to-floats to apply.
        out (Storage): storage for out tensor.
        out_shape (Shape): shape for out tensor.
        out_strides (Strides): strides for out tensor.
        in_storage (Storage): storage for in tensor.
        in_shape (Shape): shape for in tensor.
        in_strides (Strides): strides for in tensor.

    Returns:
    -------
        Tensor map function.

    """

    def _map(
        out: Storage,
        out_shape: Shape,
        out_strides: Strides,
        in_storage: Storage,
        in_shape: Shape,
        in_strides: Strides,
    ) -> None:
        # ASSIGN3.1
        # Check both stride alignment and equal shapes to avoid explicit indexing
        if (
            len(out_strides) != len(in_strides)
            or (out_strides != in_strides).any()
            or (out_shape != in_shape).any()
        ):
            # if not stride-aligned, index explicitly; run loop in parallel
            # Loop through all elements in the ordinal array (1d)
            for i in prange(len(out)):
                # Initialize all indices using numpy buffers
                out_index: Index = np.empty(MAX_DIMS, np.int32)
                in_index: Index = np.empty(MAX_DIMS, np.int32)
                # Convert an `ordinal` to an index in the `shape`
                to_index(i, out_shape, out_index)
                # Broadcast indices from out_shape to in_shape
                broadcast_index(out_index, out_shape, in_shape, in_index)
                # Converts a multidimensional tensor `index` into a single-dimensional
                # position in storage based on out_strides and in_strides
                o = index_to_position(out_index, out_strides)
                j = index_to_position(in_index, in_strides)
                # Apply fn to input value and store result in the output array
                out[o] = fn(in_storage[j])
        else:
            # Using prange for parallel loop; The main loop iterates over the output
            # tensor indices in parallel, utilizing prange to allow for parallel execution
            for i in prange(len(out)):
                out[i] = fn(in_storage[i])
        # END ASSIGN3.1

    return njit(parallel=True)(_map)


def tensor_zip(fn: Callable[[float, float], float]) -> Any:
    """NUMBA higher-order tensor zip function. See `tensor_ops.py` for description.

    Optimizations:

    * Main loop in parallel
    * All indices use numpy buffers
    * When `out`, `a`, `b` are stride-aligned, avoid indexing

    Args:
    ----
        fn: function maps two floats to float to apply.
        out (array): storage for `out` tensor.
        out_shape (array): shape for `out` tensor.
        out_strides (array): strides for `out` tensor.
        a_storage (array): storage for `a` tensor.
        a_shape (array): shape for `a` tensor.
        a_strides (array): strides for `a` tensor.
        b_storage (array): storage for `b` tensor.
        b_shape (array): shape for `b` tensor.
        b_strides (array): strides for `b` tensor.

    Returns:
    -------
        Tensor zip function.

    """

    def _zip(
        out: Storage,
        out_shape: Shape,
        out_strides: Strides,
        a_storage: Storage,
        a_shape: Shape,
        a_strides: Strides,
        b_storage: Storage,
        b_shape: Shape,
        b_strides: Strides,
    ) -> None:
        # ASSIGN3.1
        # Check both stride alignment and equal shapes to avoid explicit indexing
        if (
            len(out_strides) != len(a_strides)
            or len(out_strides) != len(b_strides)
            or (out_strides != a_strides).any()
            or (out_strides != b_strides).any()
            or (out_shape != a_shape).any()
            or (out_shape != b_shape).any()
        ):
            # if not stride-aligned, index explicitly; run loop in parallel
            # Loop through all elements in the ordinal array (1d)
            for i in prange(len(out)):
                # Initialize all indices using numpy buffers
                out_index: Index = np.empty(MAX_DIMS, np.int32)
                a_index: Index = np.empty(MAX_DIMS, np.int32)
                b_index: Index = np.empty(MAX_DIMS, np.int32)
                # Convert an `ordinal` to an index in the `shape`
                to_index(i, out_shape, out_index)
                # Converts a multidimensional tensor `index` into a
                # single-dimensional position in storage based on out_strides
                o = index_to_position(out_index, out_strides)
                # Broadcast indices from out_shape to a_shape
                broadcast_index(out_index, out_shape, a_shape, a_index)
                # Converts a multidimensional tensor `index` into a
                # single-dimensional position in storage based on a_strides
                j = index_to_position(a_index, a_strides)
                # Broadcast indices from out_shape to b_shape
                broadcast_index(out_index, out_shape, b_shape, b_index)
                # Converts a multidimensional tensor `index` into a
                # single-dimensional position in storage based on b_strides
                k = index_to_position(b_index, b_strides)
                # Apply fn to input value and store result in the output array
                out[o] = fn(a_storage[j], b_storage[k])
        else:
            # Using prange for parallel loop; The main loop iterates over the output
            # tensor indices in parallel, utilizing prange to allow for parallel execution
            for i in prange(len(out)):
                out[i] = fn(a_storage[i], b_storage[i])
        # END ASSIGN3.1

    return njit(parallel=True)(_zip)


def tensor_reduce(fn: Callable[[float, float], float]) -> Any:
    """NUMBA higher-order tensor reduce function. See `tensor_ops.py` for description.

    Optimizations:

    * Main loop in parallel
    * All indices use numpy buffers
    * Inner-loop should not call any functions or write non-local variables

    Args:
    ----
        fn: reduction function mapping two floats to float.
        out (Storage): storage for `out` tensor.
        out_shape (Shape): shape for `out` tensor.
        out_strides (Strides): strides for `out` tensor.
        a_storage (Storage): storage for `a` tensor.
        a_shape (Shape): shape for `a` tensor.
        a_strides (Strides): strides for `a` tensor.
        reduce_dim (int): dimension to reduce out

    Returns:
    -------
        Tensor reduce function

    """

    def _reduce(
        out: Storage,
        out_shape: Shape,
        out_strides: Strides,
        a_storage: Storage,
        a_shape: Shape,
        a_strides: Strides,
        reduce_dim: int,
    ) -> None:
        # ASSIGN3.1
        # Initialize the index for the output tensor
        # Loop through all elements in the ordinal array (1d)
        for i in prange(len(out)):
            # Initialize all indices using numpy buffers
            out_index: Index = np.empty(MAX_DIMS, np.int32)
            reduce_size = a_shape[reduce_dim]
            # Convert an `ordinal` to an index in the `shape`
            to_index(i, out_shape, out_index)
            # Converts a multidimensional tensor `index` into a single-dimensional
            # position in storage based on out_strides and a_strides
            o = index_to_position(out_index, out_strides)
            accum = out[o]
            j = index_to_position(out_index, a_strides)
            step = a_strides[reduce_dim]
            # Inner-loop should not call any external functions or write non-local variables
            for s in range(reduce_size):
                # call fn normally inside the inner loop
                accum = fn(accum, a_storage[j])
                j += step
            out[o] = accum
        # END ASSIGN3.1

    return njit(parallel=True)(_reduce)


def _tensor_matrix_multiply(
    out: Storage,
    out_shape: Shape,
    out_strides: Strides,
    a_storage: Storage,
    a_shape: Shape,
    a_strides: Strides,
    b_storage: Storage,
    b_shape: Shape,
    b_strides: Strides,
) -> None:
    """NUMBA tensor matrix multiply function.

    Should work for any tensor shapes that broadcast as long as

    ```
    assert a_shape[-1] == b_shape[-2]
    ```

    Optimizations:

    * Outer loop in parallel
    * No index buffers or function calls
    * Inner loop should have no global writes, 1 multiply.


    Args:
    ----
        out (Storage): storage for `out` tensor
        out_shape (Shape): shape for `out` tensor
        out_strides (Strides): strides for `out` tensor
        a_storage (Storage): storage for `a` tensor
        a_shape (Shape): shape for `a` tensor
        a_strides (Strides): strides for `a` tensor
        b_storage (Storage): storage for `b` tensor
        b_shape (Shape): shape for `b` tensor
        b_strides (Strides): strides for `b` tensor

    Returns:
    -------
        None : Fills in `out`

    """
    a_batch_stride = a_strides[0] if a_shape[0] > 1 else 0
    b_batch_stride = b_strides[0] if b_shape[0] > 1 else 0

    # ASSIGN3.2
    # Outer loop in parallel
    for i1 in prange(out_shape[0]):
        # All inner loops should have no global writes, 1 multiply.
        for i2 in prange(out_shape[1]):
            for i3 in prange(out_shape[2]):
                # Get the positions of the starting elements in the storage arrays
                a_inner = i1 * a_batch_stride + i2 * a_strides[1]
                # Get the positions of the starting elements in the storage arrays
                b_inner = i1 * b_batch_stride + i3 * b_strides[2]
                # initialize the accumulator of products of elements in
                # the row of matrix A and the column of matrix B
                acc = 0.0
                for _ in range(a_shape[2]):
                    a_inner = i1 * a_batch_stride + i2 * a_strides[1]
                    b_inner = i1 * b_batch_stride + i3 * b_strides[2]
                    acc = 0.0
                    for _ in range(a_shape[2]):
                        acc += a_storage[a_inner] * b_storage[b_inner]
                        a_inner += a_strides[2]
                        b_inner += b_strides[1]
                    # Calculate output position (i,j,k) of the current element in the output array
                    out_position = (
                        i1 * out_strides[0] + i2 * out_strides[1] + i3 * out_strides[2]
                    )
                    out[out_position] = acc
    # END ASSIGN3.2


tensor_matrix_multiply = njit(parallel=True, fastmath=True)(_tensor_matrix_multiply)
