from __future__ import annotations

from typing import TYPE_CHECKING

import minitorch

from . import operators
from .autodiff import Context

if TYPE_CHECKING:
    from typing import Tuple

    from .scalar import Scalar, ScalarLike


def wrap_tuple(x: float | Tuple[float, ...]) -> Tuple[float, ...]:
    """Turn a possible value into a tuple"""
    if isinstance(x, tuple):
        return x
    return (x,)


class ScalarFunction:
    """A wrapper for a mathematical function that processes and produces
    Scalar variables.

    This is a static class and is never instantiated. We use `class`
    here to group together the `forward` and `backward` code.
    """

    @classmethod
    def _backward(cls, ctx: Context, d_out: float) -> Tuple[float, ...]:
        return wrap_tuple(cls.backward(ctx, d_out))  # type: ignore

    @classmethod
    def _forward(cls, ctx: Context, *inps: float) -> float:
        return cls.forward(ctx, *inps)  # type: ignore

    @classmethod
    def apply(cls, *vals: ScalarLike) -> Scalar:
        """Applies the forward function to the input Scalars and records their history for backpropagation.

        Args:
        ----
            cls: Refers to class itself, the ScalarFunction class
            *vals: Scalar-like inputs (either Scalar instances or values that can be converted to Scalars)
            that the function will process.

        Returns:
        -------
            Scalar: A new Scalar that represents the result of applying the forward function to the inputs.
            The returned Scalar also contains the operation history, allowing for autodiff
            (backpropagation).

        """
        raw_vals = []
        scalars = []
        for v in vals:
            if isinstance(v, minitorch.scalar.Scalar):
                scalars.append(v)
                raw_vals.append(v.data)
            else:
                scalars.append(minitorch.scalar.Scalar(v))
                raw_vals.append(v)

        # Create the context.
        ctx = Context(False)

        # Call forward with the variables.
        c = cls._forward(ctx, *raw_vals)
        assert isinstance(c, float), "Expected return type float got %s" % (type(c))

        # Create a new variable from the result with a new history.
        back = minitorch.scalar.ScalarHistory(cls, ctx, scalars)
        return minitorch.scalar.Scalar(c, back)


# Examples
class Add(ScalarFunction):
    """Addition function $f(x, y) = x + y$"""

    @staticmethod
    def forward(ctx: Context, a: float, b: float) -> float:
        """Forward pass for addition.

        Args:
        ----
            ctx: Context object to store information for backward pass.
            a: The first operand.
            b: The second operand.

        Returns:
        -------
            The result of a + b.

        """
        return a + b

    @staticmethod
    def backward(ctx: Context, d_output: float) -> Tuple[float, ...]:
        """Backward pass for addition. Since the derivative of addition is 1 for both inputs,
        the gradients with respect to `a` and `b` are the same as `d_output`.

        Args:
        ----
            ctx: Context object that holds saved values from forward pass.
            d_output: Gradient of the output from the next layer.

        Returns:
        -------
            Gradients with respect to `a` and `b` (both equal to `d_output`).

        """
        return d_output, d_output


class Log(ScalarFunction):
    """Log function $f(x) = log(x)$"""

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        """Forward pass for the logarithm function.

        Args:
        ----
            ctx: Context object to store information for backward pass.
            a: The input value.

        Returns:
        -------
            The natural logarithm of `a` using `operators.log`.

        """
        ctx.save_for_backward(a)
        return operators.log(a)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        """Backward pass for the logarithm function. The derivative of log(x) is 1/x,
        so the gradient with respect to `a` is `d_output / a`.

        Args:
        ----
            ctx: Context object that holds saved values from forward pass.
            d_output: Gradient of the output from the next layer.

        Returns:
        -------
            The gradient with respect to `a`, i.e., d_output / a.

        """
        (a,) = ctx.saved_values
        return operators.log_back(a, d_output)


# To implement.


class Mul(ScalarFunction):
    """Multiplication function $f(x, y) = x * y$"""

    @staticmethod
    def forward(ctx: Context, a: float, b: float) -> float:
        """Forward pass for multiplication. Computes the product of `a` and `b`.

        Args:
        ----
            ctx: Context object to store information for backward pass.
            a: The first operand.
            b: The second operand.

        Returns:
        -------
            The result of a * b.

        """
        # ASSIGN1.2
        # Save the inputs for use in the backward pass.
        ctx.save_for_backward(a, b)
        c = a * b
        return c
        # END ASSIGN1.2

    @staticmethod
    def backward(ctx: Context, d_output: float) -> Tuple[float, float]:
        """Backward pass for multiplication. Computes the gradients of `a` and `b`.

        Args:
        ----
            ctx: Context object that holds saved values from forward pass.
            d_output: Gradient of the output from the next layer.

        Returns:
        -------
            Gradients with respect to `a` and `b`.

        """
        # ASSIGN1.4
        # Retrieve saved values from the forward pass.
        a, b = ctx.saved_values
        # The gradient with respect to `a` is `b * d_output` and vice versa.
        return b * d_output, a * d_output
        # END ASSIGN1.4


class Inv(ScalarFunction):
    """Inverse function $f(x) = 1 / x$"""

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        """Forward pass for inverse. Computes the inverse of `a`.

        Args:
        ----
            ctx: Context object to store information for backward pass.
            a: The input value.

        Returns:
        -------
            The result of 1 / a.

        """
        # ASSIGN1.2
        # Save the input for use in the backward pass.
        ctx.save_for_backward(a)
        return operators.inv(a)
        # END ASSIGN1.2

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        """Backward pass for inverse. Computes the gradient with respect to `a`.

        Args:
        ----
            ctx: Context object that holds saved values from forward pass.
            d_output: Gradient of the output from the next layer.

        Returns:
        -------
            The gradient with respect to `a`.

        """
        # ASSIGN1.4
        # Retrieve saved value from the forward pass.
        (a,) = ctx.saved_values
        # Gradient of 1 / a is -1 / a^2, so multiply by d_output with pre-defined inv_back operator
        return operators.inv_back(a, d_output)
        # END ASSIGN1.4


class Neg(ScalarFunction):
    """Negation function $f(x) = -x$"""

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        """Forward pass for negation. Negates the input `a`.

        Args:
        ----
            ctx: Context object to store information for backward pass.
            a: The input value.

        Returns:
        -------
            The negation of `a`, i.e., -a.

        """
        # ASSIGN1.2
        return -a
        # END ASSIGN1.2

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        """Backward pass for negation. Computes the gradient with respect to `a`.

        Args:
        ----
            ctx: Context object that holds saved values from forward pass.
            d_output: Gradient of the output from the next layer.

        Returns:
        -------
            The gradient with respect to `a`, which is -d_output.

        """
        # ASSIGN1.4
        return -d_output
        # END ASSIGN1.4


class Sigmoid(ScalarFunction):
    """Sigmoid function $f(x) = 1 / (1 + e^{-x})$"""

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        """Forward pass for sigmoid. Computes the sigmoid of `a`.

        Args:
        ----
            ctx: Context object to store information for backward pass.
            a: The input value.

        Returns:
        -------
            The result of sigmoid(a).

        """
        # ASSIGN1.2
        out = operators.sigmoid(a)
        ctx.save_for_backward(out)
        return out
        # END ASSIGN1.2

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        """Backward pass for sigmoid. Computes the gradient with respect to `a`.

        Args:
        ----
            ctx: Context object that holds saved values from forward pass.
            d_output: Gradient of the output from the next layer.

        Returns:
        -------
            The gradient with respect to `a`.

        """
        # ASSIGN1.4
        # Retrieve saved sigmoid value from forward pass.
        sigma: float = ctx.saved_values[0]
        # Gradient of sigmoid is sigmoid(a) * (1 - sigmoid(a)), so multiply by d_output.
        return sigma * (1.0 - sigma) * d_output
        # END ASSIGN1.4


class ReLU(ScalarFunction):
    """ReLU function $f(x) = max(0, x)$"""

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        """Forward pass for ReLU using `operators.relu`.

        Args:
        ----
            ctx: Context object to store information for backward pass.
            a: The input value.

        Returns:
        -------
            max(0, a).

        """
        # ASSIGN1.2
        ctx.save_for_backward(a)
        return operators.relu(a)
        # END ASSIGN1.2

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        """Backward pass for ReLU. Computes the gradient with respect to `a`.

        Args:
        ----
            ctx: Context object that holds saved values from forward pass.
            d_output: Gradient of the output from the next layer.

        Returns:
        -------
            The gradient with respect to `a`.

        """
        # ASSIGN1.4
        (a,) = ctx.saved_values
        return operators.relu_back(a, d_output)
        # END ASSIGN1.4


class Exp(ScalarFunction):
    """Exponential function $f(x) = e^x$"""

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        """Forward pass for exponential using `operators.exp`.

        Args:
        ----
            ctx: Context object to store information for backward pass.
            a: The input value.

        Returns:
        -------
            The result of exp(a).

        """
        # ASSIGN1.2
        out = operators.exp(a)  # e^a
        ctx.save_for_backward(out)
        return out
        # END ASSIGN1.2

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        """Backward pass for exponential. Computes the gradient with respect to `a`.

        Args:
        ----
            ctx: Context object that holds saved values from forward pass.
            d_output: Gradient of the output from the next layer.

        Returns:
        -------
            The gradient with respect to `a`, which is exp(a).

        """
        # ASSIGN1.4
        out: float = ctx.saved_values[0]  # derivative is (e^a)
        return d_output * out
        # END ASSIGN1.4


class LT(ScalarFunction):
    """Less-than function $f(x) =$ 1.0 if x is less than y else 0.0"""

    @staticmethod
    def forward(ctx: Context, a: float, b: float) -> float:
        """Forward pass for less than using `operators.lt`.

        Args:
        ----
            ctx: Context object to store information for backward pass.
            a: The first operand.
            b: The second operand.

        Returns:
        -------
            1.0 if a < b, else 0.0.

        """
        # ASSIGN1.2
        return 1.0 if a < b else 0.0
        # END ASSIGN1.2

    @staticmethod
    def backward(ctx: Context, d_output: float) -> Tuple[float, float]:
        """Backward pass for less than. There is no gradient for non-differentiable comparison.

        Args:
        ----
            ctx: Context object that holds saved values from forward pass.
            d_output: Gradient of the output from the next layer.

        Returns:
        -------
            Zero gradient for both inputs since this is non-differentiable.

        """
        # ASSIGN1.4
        return 0.0, 0.0
        # END ASSIGN1.4


class EQ(ScalarFunction):
    """Equal function $f(x) =$ 1.0 if x is equal to y else 0.0"""

    @staticmethod
    def forward(ctx: Context, a: float, b: float) -> float:
        """Forward pass for equality using `operators.eq`.

        Args:
        ----
            ctx: Context object to store information for backward pass.
            a: The first operand.
            b: The second operand.

        Returns:
        -------
            1.0 if a == b, else 0.0.

        """
        # ASSIGN1.2
        return 1.0 if a == b else 0.0
        # END ASSIGN1.2

    @staticmethod
    def backward(ctx: Context, d_output: float) -> Tuple[float, float]:
        """Backward pass for equality. There is no gradient for non-differentiable comparison.

        Args:
        ----
            ctx: Context object that holds saved values from forward pass.
            d_output: Gradient of the output from the next layer.

        Returns:
        -------
            A tuple of zero gradients, since the comparison is non-differentiable.

        """
        # ASSIGN1.4
        # Since equality comparison is not differentiable, return zero for both inputs.
        return 0.0, 0.0
        # END ASSIGN1.4
