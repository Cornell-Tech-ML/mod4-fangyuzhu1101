"""Implementation of the core Tensor object for autodifferentiation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from . import operators
from .autodiff import Context, Variable, backpropagate
from .tensor_data import TensorData

# Comment these out if not yet implemented
from .tensor_functions import (
    EQ,
    LT,
    Add,
    All,
    Copy,
    Exp,
    Inv,
    IsClose,
    Log,
    MatMul,
    Mul,
    Neg,
    Permute,
    ReLU,
    Sigmoid,
    Sum,
    View,
    tensor,
)

if TYPE_CHECKING:
    from typing import Any, Iterable, List, Optional, Sequence, Tuple, Type, Union

    import numpy.typing as npt

    from .tensor_data import Shape, Storage, Strides, UserIndex, UserShape, UserStrides
    from .tensor_functions import Function
    from .tensor_ops import TensorBackend

    TensorLike = Union[float, int, "Tensor"]


@dataclass
class History:
    """`History` stores the history of `Function` operations that was
    used to construct the current Variable.
    """

    last_fn: Optional[Type[Function]] = None
    ctx: Optional[Context] = None
    inputs: Sequence[Tensor] = ()


_tensor_count = 0


class Tensor:
    """Tensor is a generalization of Scalar in that it is a Variable that
    handles multidimensional arrays.
    """

    backend: TensorBackend
    history: Optional[History]
    grad: Optional[Tensor]
    _tensor: TensorData
    unique_id: int
    name: str

    def __init__(
        self,
        v: TensorData,
        back: Optional[History] = None,
        name: Optional[str] = None,
        backend: Optional[TensorBackend] = None,
    ):
        global _tensor_count
        _tensor_count += 1
        self.unique_id = _tensor_count
        assert isinstance(v, TensorData)
        assert backend is not None
        self._tensor = v
        self.history = back
        self.backend = backend
        self.grad = None
        if name is not None:
            self.name = name
        else:
            self.name = str(self.unique_id)

        self.f = backend

    def requires_grad_(self, x: bool) -> None:
        """Sets the tensor to track gradients for backpropagation if `x` is True."""
        self.history = History()

    def requires_grad(self) -> bool:
        """Returns whether the tensor is tracking gradients for backpropagation."""
        return self.history is not None

    def to_numpy(self) -> npt.NDArray[np.float64]:
        """Returns
        Converted to numpy array

        """
        return self.contiguous()._tensor._storage.reshape(self.shape)

    def _ensure_tensor(self, b: TensorLike) -> Tensor:
        """Turns a python number into a tensor with the same backend."""
        if isinstance(b, (int, float)):
            c = Tensor.make([b], (1,), backend=self.backend)
        else:
            b._type_(self.backend)
            c = b
        return c

    def item(self) -> float:
        """Convert a 1-element tensor to a float"""
        assert self.size == 1
        x: float = self._tensor._storage[0]
        return x

    def contiguous(self) -> Tensor:
        """Return a contiguous tensor with the same data"""
        return Copy.apply(self)

    def __repr__(self) -> str:
        return self._tensor.to_string()

    def __getitem__(self, key: Union[int, UserIndex]) -> float:
        key2 = (key,) if isinstance(key, int) else key
        return self._tensor.get(key2)

    def __setitem__(self, key: Union[int, UserIndex], val: float) -> None:
        key2 = (key,) if isinstance(key, int) else key
        self._tensor.set(key2, val)

    # Internal methods used for autodiff.
    def _type_(self, backend: TensorBackend) -> None:
        self.backend = backend
        if backend.cuda:  # pragma: no cover
            self._tensor.to_cuda_()

    def _new(self, tensor_data: TensorData) -> Tensor:
        return Tensor(tensor_data, backend=self.backend)

    @staticmethod
    def make(
        storage: Union[Storage, List[float]],
        shape: UserShape,
        strides: Optional[UserStrides] = None,
        backend: Optional[TensorBackend] = None,
    ) -> Tensor:
        """Create a new tensor from data"""
        return Tensor(TensorData(storage, shape, strides), backend=backend)

    def expand(self, other: Tensor) -> Tensor:
        """Method used to allow for backprop over broadcasting.
        This method is called when the output of `backward`
        is a different size than the input of `forward`.


        Args:
        ----
            other : backward tensor (must broadcast with self)

        Returns:
        -------
            Expanded version of `other` with the right derivatives

        """
        # Case 1: Both the same shape.
        if self.shape == other.shape:
            return other

        # Case 2: Backward is a smaller than self. Broadcast up.
        true_shape = TensorData.shape_broadcast(self.shape, other.shape)
        buf = self.zeros(true_shape)
        self.backend.id_map(other, buf)
        if self.shape == true_shape:
            return buf

        # Case 3: Still different, reduce extra dims.
        out = buf
        orig_shape = [1] * (len(out.shape) - len(self.shape)) + list(self.shape)
        for dim, shape in enumerate(out.shape):
            if orig_shape[dim] == 1 and shape != 1:
                out = self.backend.add_reduce(out, dim)
        assert out.size == self.size, f"{out.shape} {self.shape}"
        # START CODE CHANGE (2021)
        return Tensor.make(out._tensor._storage, self.shape, backend=self.backend)
        # END CODE CHANGE (2021)

    def zeros(self, shape: Optional[UserShape] = None) -> Tensor:
        """Returns a new tensor filled with zeros, matching the current tensor's shape or the provided shape."""

        def zero(shape: UserShape) -> Tensor:
            return Tensor.make(
                [0.0] * int(operators.prod(shape)), shape, backend=self.backend
            )

        if shape is None:
            out = zero(self.shape)
        else:
            out = zero(shape)
        out._type_(self.backend)
        return out

    def tuple(self) -> Tuple[Storage, Shape, Strides]:
        """Get the tensor data info as a tuple."""
        return self._tensor.tuple()

    def detach(self) -> Tensor:
        """Detach from backprop"""
        return Tensor(self._tensor, backend=self.backend)

    # Variable elements for backprop

    def accumulate_derivative(self, x: Any) -> None:
        """Add `val` to the the derivative accumulated on this variable.
        Should only be called during autodifferentiation on leaf variables.

        Args:
        ----
            x : value to be accumulated

        """
        assert self.is_leaf(), "Only leaf variables can have derivatives."
        if self.grad is None:
            self.grad = Tensor.make(
                [0.0] * int(operators.prod(self.shape)),
                self.shape,
                backend=self.backend,
            )
        self.grad += x

    def is_leaf(self) -> bool:
        """True if this variable created by the user (no `last_fn`)"""
        return self.history is not None and self.history.last_fn is None

    def is_constant(self) -> bool:
        """Checks if the tensor is a constant (i.e., not tracking gradients)."""
        return self.history is None

    @property
    def parents(self) -> Iterable[Variable]:
        """Returns the input tensors (parents) involved in the operation that produced this tensor."""
        assert self.history is not None
        return self.history.inputs

    def chain_rule(self, d_output: Any) -> Iterable[Tuple[Variable, Any]]:
        """Applies the chain rule to compute the gradients for the input tensors (parents)."""
        h = self.history
        assert h is not None
        assert h.last_fn is not None
        assert h.ctx is not None

        x = h.last_fn._backward(h.ctx, d_output)
        assert len(x) == len(h.inputs), f"Bug in function {h.last_fn}"
        return [
            (inp, inp.expand(self._ensure_tensor(d_in)))
            for inp, d_in in zip(h.inputs, x)
        ]

    def backward(self, grad_output: Optional[Tensor] = None) -> None:
        """Performs backpropagation on the tensor, computing gradients for all inputs."""
        if grad_output is None:
            assert self.shape == (1,), "Must provide grad_output if non-scalar"
            grad_output = Tensor.make([1.0], (1,), backend=self.backend)
        backpropagate(self, grad_output)

    def __truediv__(self, b: TensorLike) -> Tensor:
        return Mul.apply(self, Inv.apply(self._ensure_tensor(b)))

    def __rtruediv__(self, b: TensorLike) -> Tensor:
        return Mul.apply(self._ensure_tensor(b), Inv.apply(self))

    def __matmul__(self, b: Tensor) -> Tensor:
        """Not used until Module 3"""
        return MatMul.apply(self, b)

    @property
    def shape(self) -> UserShape:
        """Returns
        shape of the tensor

        """
        return self._tensor.shape

    # Functions
    # TODO: Implement for Task 2.3.
    @property
    def dims(self) -> int:
        """Get the number of dimensions (rank) of the tensor.

        Returns
        -------
            int: dimensionality of the tensor

        """
        return self._tensor.dims  # Number of dimensions in the shape

    @property
    def size(self) -> int:
        """Get the total number of elements in the tensor.

        Returns
        -------
            int: size of the tensor

        """
        return self._tensor.size  # Multiply all dimensions together

    # Functions
    def __add__(self, b: TensorLike) -> Tensor:
        """Perform element-wise addition between this tensor and another value.

        Args:
        ----
            b (TensorLike): The value to add. Can be a scalar or another tensor.

        Returns:
        -------
            Tensor: A new tensor resulting from element-wise addition.

        """
        return Add.apply(self, self._ensure_tensor(b))

    def __sub__(self, b: TensorLike) -> Tensor:
        """Perform element-wise subtraction between this tensor and another value.

        Args:
        ----
            b (TensorLike): The value to subtract. Can be a scalar or another tensor.

        Returns:
        -------
            Tensor: A new tensor resulting from element-wise subtraction.

        """
        return Add.apply(self, -self._ensure_tensor(b))

    def __mul__(self, b: TensorLike) -> Tensor:
        """Perform element-wise multiplication between this tensor and another value.

        Args:
        ----
            b (TensorLike): The value to multiply by. Can be a scalar or another tensor.

        Returns:
        -------
            Tensor: A new tensor resulting from element-wise multiplication.

        """
        return Mul.apply(self, self._ensure_tensor(b))

    def __lt__(self, b: TensorLike) -> Tensor:
        """Perform element-wise less-than comparison between this tensor and another value.

        Args:
        ----
            b (TensorLike): The value to compare against. Can be a scalar or another tensor.

        Returns:
        -------
            Tensor: A new tensor with 1 where the comparison is true, and 0 where false.

        """
        return LT.apply(self, self._ensure_tensor(b))

    def __eq__(self, b: TensorLike) -> Tensor:  # type: ignore[override]
        """Perform element-wise equality comparison between this tensor and another value.

        Args:
        ----
            b (TensorLike): The value to compare against. Can be a scalar or another tensor.

        Returns:
        -------
            Tensor: A new tensor with 1 where the values are equal, and 0 where not.

        """
        return EQ.apply(self, self._ensure_tensor(b))

    def __gt__(self, b: TensorLike) -> Tensor:
        """Perform element-wise greater-than comparison between this tensor and another value.

        Args:
        ----
            b (TensorLike): The value to compare against. Can be a scalar or another tensor.

        Returns:
        -------
            Tensor: A new tensor with 1 where the comparison is true, and 0 where false.

        """
        return LT.apply(
            self._ensure_tensor(b), self
        )  # `b < self` is equivalent to `self > b`

    def __neg__(self) -> Tensor:
        """Perform element-wise negation of this tensor.

        Returns
        -------
            Tensor: A new tensor where each element is the negation of the original tensor.

        """
        return Neg.apply(self)

    def __radd__(self, b: TensorLike) -> Tensor:
        """Perform element-wise addition where this tensor is on the right-hand side.

        Args:
        ----
            b (TensorLike): The value to add to this tensor. Can be a scalar or another tensor.

        Returns:
        -------
            Tensor: A new tensor resulting from element-wise addition.

        """
        return self + b  # Reverse the order for right addition

    def __rmul__(self, b: TensorLike) -> Tensor:
        """Perform element-wise multiplication where this tensor is on the right-hand side.

        Args:
        ----
            b (TensorLike): The value to multiply with this tensor. Can be a scalar or another tensor.

        Returns:
        -------
            Tensor: A new tensor resulting from element-wise multiplication.

        """
        return self * b  # Reverse the order for right multiplication

    def all(self, dim: Optional[int] = None) -> Tensor:
        """Check if all elements of the tensor (or along a dimension) are non-zero.

        Args:
        ----
            dim (Optional[int]): The dimension to check along. If None, check all elements.

        Returns:
        -------
            Tensor: A tensor with value 1 if all elements are non-zero, otherwise 0.

        """
        if dim is None:
            return All.apply(self.view(self.size), self._ensure_tensor(0))
        else:
            return All.apply(self, self._ensure_tensor(dim))

    def is_close(self, y: Tensor) -> Tensor:
        """Apply the is_close function element-wise to this tensor.

        Args:
        ----
            y: The tensor to compare against.

        Returns:
        -------
            Tensor: A tensor with value 1 where the elements are close, otherwise 0.

        """
        return IsClose.apply(self, y)

    def sigmoid(self) -> Tensor:
        """Apply the sigmoid function element-wise to this tensor.

        Returns
        -------
            Tensor: A new tensor with the sigmoid applied to each element.

        """
        return Sigmoid.apply(self)

    def relu(self) -> Tensor:
        """Apply the ReLU (Rectified Linear Unit) function element-wise to this tensor.

        Returns
        -------
            Tensor: A new tensor with ReLU applied to each element.

        """
        return ReLU.apply(self)

    def log(self) -> Tensor:
        """Apply the natural logarithm function element-wise to this tensor.

        Returns
        -------
            Tensor: A new tensor with the logarithm of each element.

        """
        return Log.apply(self)

    def exp(self) -> Tensor:
        """Apply the exponential function element-wise to this tensor.

        Returns
        -------
            Tensor: A new tensor with the exponential of each element.

        """
        return Exp.apply(self)

    def sum(self, dim: Optional[int] = None) -> Tensor:
        """Sum all elements of the tensor or along a specific dimension.
        Compute the sum over dimension `dim`.

        Args:
        ----
            dim (Optional[int]): The dimension to sum along. If None, sum all elements.

        Returns:
        -------
            Tensor: A new tensor with the sum of the elements.

        """
        if dim is None:
            # If dim is None, sum over all elements in the tensor
            return Sum.apply(self.contiguous().view(self.size), self._ensure_tensor(0))
        else:
            # Sum along the specific dimension
            return Sum.apply(self, self._ensure_tensor(dim))

    def mean(self, dim: Optional[int] = None) -> Tensor:
        """Compute the mean of all elements or along a specific dimension.
        Compute the mean over dimension `dim`.

        Args:
        ----
            dim (Optional[int]): The dimension to compute the mean along. If None, compute mean of all elements.

        Returns:
        -------
            Tensor: A new tensor with the mean of the elements.

        """
        if dim is not None:
            # Compute mean over all elements
            return self.sum(dim) / self.shape[dim]
        else:
            # Compute mean along the specified dimension
            return self.sum() / self.size

    def permute(self, *order: int) -> Tensor:
        """Permute (rearrange) the dimensions of the tensor.
        Permute tensor dimensions to *order.

        Args:
        ----
            order (int): The new ordering of the dimensions.

        Returns:
        -------
            Tensor: A new tensor with permuted dimensions.

        """
        # Create a tensor for the order and apply the View function
        return Permute.apply(self, tensor(list(order)))

    def view(self, *shape: int) -> Tensor:
        """Reshape the tensor without changing its data.
        Change the shape of the tensor to a new shape with the same size.

        Args:
        ----
            shape (int): The new shape.

        Returns:
        -------
            Tensor: A new tensor with the specified shape.

        """
        # Create a tensor for the new shape
        return View.apply(self, tensor(list(shape)))

    def zero_grad_(self) -> None:  # pragma: no cover
        """Reset the derivative on this variable."""
        self.grad = None
