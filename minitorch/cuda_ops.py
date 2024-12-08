# type: ignore
# Currently pyright doesn't support numba.cuda

from typing import Any, Callable, Optional

import numba
from numba import cuda

from .tensor import Tensor
from .tensor_data import (
    MAX_DIMS,
    Shape,
    Storage,
    Strides,
    TensorData,
    broadcast_index,
    index_to_position,
    shape_broadcast,
    to_index,
)
from .tensor_ops import MapProto, TensorOps

# This code will CUDA compile fast versions your tensor_data functions.
# If you get an error, read the docs for NUMBA as to what is allowed
# in these functions.
to_index = cuda.jit(device=True)(to_index)
index_to_position = cuda.jit(device=True)(index_to_position)
broadcast_index = cuda.jit(device=True)(broadcast_index)

THREADS_PER_BLOCK = 32


class CudaOps(TensorOps):
    cuda = True

    @staticmethod
    def map(fn: Callable[[float], float]) -> MapProto:
        """See `tensor_ops.py`"""
        f = tensor_map(cuda.jit(device=True)(fn))

        def ret(a: Tensor, out: Optional[Tensor] = None) -> Tensor:
            if out is None:
                out = a.zeros(a.shape)

            # Instantiate and run the cuda kernel.
            threadsperblock = THREADS_PER_BLOCK
            blockspergrid = (out.size + THREADS_PER_BLOCK - 1) // THREADS_PER_BLOCK
            f[blockspergrid, threadsperblock](*out.tuple(), out.size, *a.tuple())  # type: ignore
            return out

        return ret

    @staticmethod
    def zip(fn: Callable[[float, float], float]) -> Callable[[Tensor, Tensor], Tensor]:
        """See `tensor_ops.py`"""
        f = tensor_zip(cuda.jit(device=True)(fn))

        def ret(a: Tensor, b: Tensor) -> Tensor:
            c_shape = shape_broadcast(a.shape, b.shape)
            out = a.zeros(c_shape)
            threadsperblock = THREADS_PER_BLOCK
            blockspergrid = (out.size + (threadsperblock - 1)) // threadsperblock
            f[blockspergrid, threadsperblock](  # type: ignore
                *out.tuple(), out.size, *a.tuple(), *b.tuple()
            )
            return out

        return ret

    @staticmethod
    def reduce(
        fn: Callable[[float, float], float], start: float = 0.0
    ) -> Callable[[Tensor, int], Tensor]:
        """See `tensor_ops.py`"""
        f = tensor_reduce(cuda.jit(device=True)(fn))

        def ret(a: Tensor, dim: int) -> Tensor:
            out_shape = list(a.shape)
            out_shape[dim] = (a.shape[dim] - 1) // 1024 + 1
            out_a = a.zeros(tuple(out_shape))

            threadsperblock = 1024
            blockspergrid = out_a.size
            f[blockspergrid, threadsperblock](  # type: ignore
                *out_a.tuple(), out_a.size, *a.tuple(), dim, start
            )

            return out_a

        return ret

    @staticmethod
    def matrix_multiply(a: Tensor, b: Tensor) -> Tensor:
        """Matrix multiply specifically for CUDA."""
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

        # One block per batch, extra rows, extra col
        blockspergrid = (
            (out.shape[1] + (THREADS_PER_BLOCK - 1)) // THREADS_PER_BLOCK,
            (out.shape[2] + (THREADS_PER_BLOCK - 1)) // THREADS_PER_BLOCK,
            out.shape[0],
        )
        threadsperblock = (THREADS_PER_BLOCK, THREADS_PER_BLOCK, 1)

        tensor_matrix_multiply[blockspergrid, threadsperblock](
            *out.tuple(), out.size, *a.tuple(), *b.tuple()
        )

        # Undo 3d if we added it.
        if both_2d:
            out = out.view(out.shape[1], out.shape[2])
        return out


# Implement


def tensor_map(fn: Callable[[float], float]) -> Any:
    """CUDA higher-order tensor map function. ::

      fn_map = tensor_map(fn)
      fn_map(out, ... )

    Args:
    ----
        fn: function mappings floats-to-floats to apply.
        out (Storage): storage for out tensor.
        out_shape (Shape): shape for out tensor.
        out_strides (Strides): strides for out tensor.
        out_size (int): size for out tensor.
        in_storage (Storage): storage for in tensor.
        in_shape (Shape): shape for in tensor.
        in_strides (Strides): strides for in tensor.

    Returns:
    -------
        None : Fills in `out`
        Tensor map function.

    """

    def _map(
        out: Storage,
        out_shape: Shape,
        out_strides: Strides,
        out_size: int,
        in_storage: Storage,
        in_shape: Shape,
        in_strides: Strides,
    ) -> None:
        # ASSIGN3.3
        out_index = cuda.local.array(MAX_DIMS, numba.int32)
        in_index = cuda.local.array(MAX_DIMS, numba.int32)
        i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
        # Check if the thread is within the bounds or not
        if i < out_size:
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
        # END ASSIGN3.3

    return cuda.jit()(_map)


def tensor_zip(fn: Callable[[float, float], float]) -> Any:
    """CUDA higher-order tensor zipWith (or map2) function ::

      fn_zip = tensor_zip(fn)
      fn_zip(out, ...)

    Args:
    ----
        fn: function mappings two floats to float to apply.
        out (array): storage for `out` tensor.
        out_shape (array): shape for `out` tensor.
        out_strides (array): strides for `out` tensor.
        out_size (array): size for `out` tensor.
        a_storage (array): storage for `a` tensor.
        a_shape (array): shape for `a` tensor.
        a_strides (array): strides for `a` tensor.
        b_storage (array): storage for `b` tensor.
        b_shape (array): shape for `b` tensor.
        b_strides (array): strides for `b` tensor.

    Returns:
    -------
        None : Fills in `out`
        Tensor zip function.

    """

    def _zip(
        out: Storage,
        out_shape: Shape,
        out_strides: Strides,
        out_size: int,
        a_storage: Storage,
        a_shape: Shape,
        a_strides: Strides,
        b_storage: Storage,
        b_shape: Shape,
        b_strides: Strides,
    ) -> None:
        # ASSIGN3.3
        out_index = cuda.local.array(MAX_DIMS, numba.int32)
        a_index = cuda.local.array(MAX_DIMS, numba.int32)
        b_index = cuda.local.array(MAX_DIMS, numba.int32)
        i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x

        if i < out_size:
            # Convert an `ordinal` to an index in the `shape`
            to_index(i, out_shape, out_index)
            o = index_to_position(out_index, out_strides)
            broadcast_index(out_index, out_shape, a_shape, a_index)
            j = index_to_position(a_index, a_strides)
            broadcast_index(out_index, out_shape, b_shape, b_index)
            k = index_to_position(b_index, b_strides)
            out[o] = fn(a_storage[j], b_storage[k])
        # END ASSIGN3.3

    return cuda.jit()(_zip)


def _sum_practice(out: Storage, a: Storage, size: int) -> None:
    r"""Apply a practice sum kernel to prepare for reduce.

    Given an array of length $n$ and out of size $n // \text{blockDIM}$
    it should sum up each blockDim values into an out cell.

    $[a_1, a_2, ..., a_{100}]$

    |

    $[a_1 +...+ a_{31}, a_{32} + ... + a_{64}, ... ,]$

    Note: Each block must do the sum using shared memory!

    Args:
    ----
        out (Storage): storage for `out` tensor.
        a (Storage): storage for `a` tensor.
        size (int):  length of a.

    """
    BLOCK_DIM = 32
    # ASSIGN3.3
    # Shared memory allocation for the current block
    cache = cuda.shared.array(BLOCK_DIM, numba.float64)
    # Calculate the global index of the current thread
    i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    # Get the thread's local position within the block
    pos = cuda.threadIdx.x

    # Load data into shared memory from storage `a`` if within the bounds,
    # or paddling with the value of 0.0 to the reduction if out of the bounds
    if i < size:
        val = float(a[i])
        cache[pos] = val
        # Synchronize the threads to make sure that everything is loaded into cache
        cuda.syncthreads()
    else:
        cache[pos] = 0.0

    if i < size:
        # Perform the reduction within each block using a
        # doubling stride pattern with [1, 2, 4, 8, 16]
        # Each iteration halves the number of active threads
        for j in [1, 2, 4, 8, 16]:
            # Stride doubles after each iteration for [2^0=1, 2^1=2, 2^2=4, 2^3=8, 2^4=16]
            if pos % (j * 2) == 0:
                cache[pos] += cache[pos + j]
                # Synchronizes all threads within the block, ensuring that each
                # step of the reduction completes before moving to the next.
                cuda.syncthreads()
        # Store the block result from the very first thread within each block after reduction completes
        if pos == 0:  # the first thread in each block
            # Each block writes a single result to out, where out contains the
            # partial sums from each block after kernel execution
            out[cuda.blockIdx.x] = cache[0]
    # END ASSIGN3.3


jit_sum_practice = cuda.jit()(_sum_practice)


def sum_practice(a: Tensor) -> TensorData:
    """Apply a practice sum function to prepare for reduce."""
    (size,) = a.shape
    threadsperblock = THREADS_PER_BLOCK
    blockspergrid = (size // THREADS_PER_BLOCK) + 1
    out = TensorData([0.0 for i in range(2)], (2,))
    out.to_cuda_()
    jit_sum_practice[blockspergrid, threadsperblock](
        out.tuple()[0], a._tensor._storage, size
    )
    return out


def tensor_reduce(fn: Callable[[float, float], float]) -> Any:
    """CUDA higher-order tensor reduce function.

    Args:
    ----
        fn: reduction function maps two floats to float.
        out (Storage): storage for `out` tensor.
        out_shape (Shape): shape for `out` tensor.
        out_strides (Strides): strides for `out` tensor.
        out_size (int): size for `out` tensor.
        a_storage (Storage): storage for `a` tensor.
        a_shape (Shape): shape for `a` tensor.
        a_strides (Strides): strides for `a` tensor.
        reduce_dim (int): dimension to reduce out

    Returns:
    -------
        Tensor reduce function.

    """

    def _reduce(
        out: Storage,
        out_shape: Shape,
        out_strides: Strides,
        out_size: int,
        a_storage: Storage,
        a_shape: Shape,
        a_strides: Strides,
        reduce_dim: int,
        reduce_value: float,
    ) -> None:
        BLOCK_DIM = 1024
        # ASSIGN3.3
        cache = cuda.shared.array(BLOCK_DIM, numba.float64)
        out_index = cuda.local.array(MAX_DIMS, numba.int32)
        out_pos = cuda.blockIdx.x
        pos = cuda.threadIdx.x
        # Initialize shared memory with the reduction initial value
        cache[pos] = reduce_value

        if out_pos < out_size:
            # Map out_pos to the appropriate multi-dimensional index
            to_index(out_pos, out_shape, out_index)
            o = index_to_position(out_index, out_strides)
            # Now we know where we are going. (haven't used thread)

            # Adjust out_index for the reduction dimension
            out_index[reduce_dim] = out_index[reduce_dim] * BLOCK_DIM + pos
            # Check whether within bounds for the reduction dimension
            if out_index[reduce_dim] < a_shape[reduce_dim]:
                # Calculate the input position index
                in_a = index_to_position(out_index, a_strides)
                # Load the relevant value from a_storage into shared memory
                cache[pos] = a_storage[in_a]
                # Synchronize threads to ensure all threads have loaded their values into cache
                cuda.syncthreads()
                x = 0
                # Perform parallel reduction in shared memory using a binary reduction pattern [2^0=1, 2^1=2, 2^2=4, 2^3=8, ]
                while 2**x < BLOCK_DIM:
                    j = 2**x
                    if pos % (j * 2) == 0:
                        cache[pos] = fn(cache[pos], cache[pos + j])
                        # Synchronize threads to ensure each reduction step is complete
                        cuda.syncthreads()
                    x += 1
            # Only the first thread writes the reduced result to the output array
            if pos == 0:
                out[o] = cache[0]
        # END ASSIGN3.3

    return cuda.jit()(_reduce)


def _mm_practice(out: Storage, a: Storage, b: Storage, size: int) -> None:
    r"""Apply a practice square MM kernel to prepare for matmul.

    Given a storage `out` and two storage `a` and `b`. Where we know
    both are shape [size, size] with strides [size, 1].

    Size is always < 32.

    Requirements:

    * All data must be first moved to shared memory.
    * Only read each cell in `a` and `b` once.
    * Only write to global memory once per kernel.

    Compute

    ```
     for i:
         for j:
              for k:
                  out[i, j] += a[i, k] * b[k, j]
    ```

    Args:
    ----
        out (Storage): storage for `out` tensor.
        a (Storage): storage for `a` tensor.
        b (Storage): storage for `b` tensor.
        size (int): size of the square

    """
    BLOCK_DIM = 32
    # ASSIGN3.3
    # Shared memory allocation for matrices a and b.
    # Define shared memory for tiles of `a` and `b`
    a_shared = cuda.shared.array((BLOCK_DIM, BLOCK_DIM), numba.float64)
    b_shared = cuda.shared.array((BLOCK_DIM, BLOCK_DIM), numba.float64)

    # Get the indices of the threads
    i = cuda.threadIdx.x
    j = cuda.threadIdx.y

    if i >= size or j >= size:
        return

    # Load data into shared memory if within bounds
    # Given a storage out and two storage a and b; both are shape [size, size] with strides [size, 1].
    # Size is always < 32; if x_pos < size and y_pos < size:
    a_shared[i, j] = a[size * i + j]
    b_shared[i, j] = b[size * i + j]
    # Explicitly waiting after loading both shared memory `a` and `b` from one global reads
    cuda.syncthreads()  # Synchronize to ensure all threads finish loading

    # Perform the matrix multiplication
    accum = 0.0
    # Iterate over shared memory to perform computations
    for k in range(size):
        accum += a_shared[i, k] * b_shared[k, j]

    # Write the final result back to global memory
    out[size * i + j] = accum
    # END ASSIGN3.3


jit_mm_practice = cuda.jit()(_mm_practice)


def mm_practice(a: Tensor, b: Tensor) -> TensorData:
    """Performs matrix multiplication of two tensors using CUDA."""
    (size, _) = a.shape
    threadsperblock = (THREADS_PER_BLOCK, THREADS_PER_BLOCK)
    blockspergrid = 1
    out = TensorData([0.0 for i in range(size * size)], (size, size))
    out.to_cuda_()
    jit_mm_practice[blockspergrid, threadsperblock](
        out.tuple()[0], a._tensor._storage, b._tensor._storage, size
    )
    return out


def _tensor_matrix_multiply(
    out: Storage,
    out_shape: Shape,
    out_strides: Strides,
    out_size: int,
    a_storage: Storage,
    a_shape: Shape,
    a_strides: Strides,
    b_storage: Storage,
    b_shape: Shape,
    b_strides: Strides,
) -> None:
    """CUDA tensor matrix multiply function.

    Requirements:

    * All data must be first moved to shared memory.
    * Only read each cell in `a` and `b` once.
    * Only write to global memory once per kernel.

    Should work for any tensor shapes that broadcast as long as ::

    ```python
    assert a_shape[-1] == b_shape[-2]
    ```

    Args:
    ----
        out (Storage): storage for `out` tensor
        out_shape (Shape): shape for `out` tensor
        out_strides (Strides): strides for `out` tensor
        out_size (int): size for `out` tensor.
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
    BLOCK_DIM = 32
    # Code Plan:
    # 1) Move across shared dimension by block dim.
    #    a) Copy into shared memory for a matrix.
    #    b) Copy into shared memory for b matrix
    #    c) Compute the dot produce for position c[i, j]

    # ASSIGN3.4
    a_shared = cuda.shared.array((BLOCK_DIM, BLOCK_DIM), numba.float64)
    b_shared = cuda.shared.array((BLOCK_DIM, BLOCK_DIM), numba.float64)

    # Batch dimension - fixed
    batch = cuda.blockIdx.z
    # The final position c[i, j]
    i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    j = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y
    # The local position in the block.
    pi = cuda.threadIdx.x
    pj = cuda.threadIdx.y

    # Initialize accumulator
    accum = 0.0
    # Number of chunks to cover the K dimension; for example if a_shape[2] = 128,
    # it will only have 4 chunks with [0, 32, 64, 96] without the boundary of 128 (ceiling function)
    # Loop over all shared memory blocks along the shared dimension; shared_dim_start
    # is the starting point for the current block in the shared dimension;
    # It iterates from 0 to a_shape[2] (the size of the shared dimension) in steps of BLOCK_DIM
    for k_start in range(0, a_shape[2], BLOCK_DIM):
        # Load elements from `a` into shared memory
        k = k_start + pj
        if i < a_shape[1] and k < a_shape[2]:
            a_shared[pi, pj] = a_storage[
                a_batch_stride * batch + a_strides[1] * i + a_strides[2] * k
            ]
        # Load elements from `b` into shared memory
        k = k_start + pi
        if j < b_shape[2] and k < b_shape[1]:
            b_shared[pi, pj] = b_storage[
                b_batch_stride * batch + b_strides[1] * k + b_strides[2] * j
            ]
        # Synchronize threads within the block to wait for loading all shared memory from global reads
        cuda.syncthreads()

        # Perform matrix multiplication for the current block within the shared dimensions
        for k in range(BLOCK_DIM):
            # Calculate the global index for the current block in the shared dimension by adding
            # the starting point and shared_dim_index_local, which is the local offset within the block
            # Check if the global index is within bounds of the shared dimension
            if (k_start + k) < a_shape[2]:
                # Accumulate the product of corresponding elements from shared memory
                accum += a_shared[pi, k] * b_shared[k, pj]

    # Write the result back to global memory
    if i < out_shape[1] and j < out_shape[2]:
        out[out_strides[0] * batch + out_strides[1] * i + out_strides[2] * j] = accum
    # END ASSIGN3.4


tensor_matrix_multiply = cuda.jit(_tensor_matrix_multiply)
