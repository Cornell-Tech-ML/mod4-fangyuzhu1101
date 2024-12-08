from typing import Sequence

from .module import Parameter
from .scalar import Scalar


class Optimizer:
    def __init__(self, parameters: Sequence[Parameter]):
        self.parameters = parameters


class SGD(Optimizer):
    def __init__(self, parameters: Sequence[Parameter], lr: float = 1.0):
        super().__init__(parameters)
        self.lr = lr

    def zero_grad(self) -> None:
        """Resets the gradients or derivatives of all parameters to `None`, which is typically called before
        a new round of backpropagation to ensure that gradients do not accumulate across training iterations.

        Args:
        ----
            self: The instance of the SGD optimizer containing the parameters to be reset.

        Returns:
        -------
            None: This function does not return a value, it modifies the gradients of the parameters in-place.

        """
        for p in self.parameters:
            if p.value is None:
                continue
            if hasattr(p.value, "derivative"):
                if p.value.derivative is not None:
                    p.value.derivative = None
            if hasattr(p.value, "grad"):
                if p.value.grad is not None:
                    p.value.grad = None

    def step(self) -> None:
        """Updates all parameters by subtracting the learning rate multiplied
        by their gradient or derivative (a single optimization step).

        Args:
        ----
            self: The instance of the SGD optimizer containing the parameters to be updated.

        Returns:
        -------
            None: This function does not return a value, it updates the parameters in-place.

        """
        for p in self.parameters:
            if p.value is None:
                continue
            if hasattr(p.value, "derivative"):
                if p.value.derivative is not None:
                    p.update(Scalar(p.value.data - self.lr * p.value.derivative))
            elif hasattr(p.value, "grad"):
                if p.value.grad is not None:
                    p.update(p.value - self.lr * p.value.grad)
