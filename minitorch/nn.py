from typing import Tuple

from . import operators
from .autodiff import Context
from .fast_ops import FastOps
from .tensor import Tensor
from .tensor_functions import Function, rand


# List of functions in this file:
# - avgpool2d: Tiled average pooling 2D
# - argmax: Compute the argmax as a 1-hot tensor
# - Max: New Function for max operator
# - max: Apply max reduction
# - softmax: Compute the softmax as a tensor
# - logsoftmax: Compute the log of the softmax as a tensor - See https://en.wikipedia.org/wiki/LogSumExp#log-sum-exp_trick_for_log-domain_calculations
# - maxpool2d: Tiled max pooling 2D
# - dropout: Dropout positions based on random noise, include an argument to turn off


def tile(input: Tensor, kernel: Tuple[int, int]) -> Tuple[Tensor, int, int]:
    """Reshape an image tensor for 2D pooling.

    Args:
    ----
        input: batch x channel x height x width
        kernel: height x width of pooling

    Returns:
    -------
        Tensor of size batch x channel x new_height x new_width x (kernel_height * kernel_width) as well as the new_height and new_width value.

    """
    batch, channel, height, width = input.shape
    kh, kw = kernel
    assert height % kh == 0
    assert width % kw == 0
    # TODO: Implement for Task 4.3.
    new_height = height // kh
    new_width = width // kw
    output = input.contiguous().view(batch, channel, height, new_width, kw)
    output = output.permute(0, 1, 3, 2, 4)
    # batch x channel x new_height x new_width x (kernel_height * kernel_width)
    output = output.contiguous().view(batch, channel, new_height, new_width, (kh * kw))

    return output, new_height, new_width


# TODO: Implement for Task 4.3.
# - avgpool2d: Tiled average pooling 2D
def avgpool2d(input: Tensor, kernel: Tuple[int, int]) -> Tensor:
    """Tiled average pooling 2D.

    Args:
    ----
        input : batch x channel x height x width
        kernel : height x width of pooling

    Returns:
    -------
        Tensor : the averagely pooled tensor

    """
    batch, channel, height, width = input.shape
    input, new_height, new_width = tile(input, kernel)
    output = input.mean(dim=4)
    return output.view(batch, channel, new_height, new_width)


# TODO: Implement for Task 4.4.
max_reduce = FastOps.reduce(operators.max, -1e9)


# - argmax: Compute the argmax as a 1-hot tensor
def argmax(input: Tensor, dim: int) -> Tensor:
    """Compute the argmax as a 1-hot tensor.

    Args:
    ----
        input : input tensor
        dim : dimension to apply argmax


    Returns:
    -------
        Tensor : the tensor with 1 on highest cell in dim, 0 otherwise

    """
    output = max_reduce(input, dim)
    return input == output


# - Max: New Function for max operator
class Max(Function):
    @staticmethod
    def forward(ctx: Context, input: Tensor, dim: Tensor) -> Tensor:
        """Forward of max would be max reduction."""
        ctx.save_for_backward(input, int(dim.item()))
        return max_reduce(input, int(dim.item()))

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, float]:
        """Backward of max would be argmax."""
        (input, dim) = ctx.saved_values
        # dim should be an int
        return (argmax(input, dim=dim) * grad_output), dim


# - max: Apply max reduction
def max(input: Tensor, dim: int) -> Tensor:
    """Apply max reduction."""
    return Max.apply(input, input._ensure_tensor(dim))


# - softmax: Compute the softmax as a tensor
def softmax(input: Tensor, dim: int) -> Tensor:
    """Compute the softmax as a tensor.

    Args:
    ----
        input : input tensor
        dim : dimension to apply softmax

    Returns:
    -------
        Tensor : the softmax tensor

    """
    exp_input = input.exp()
    output = exp_input / exp_input.sum(dim=dim)
    return output


# - logsoftmax: Compute the log of the softmax as a tensor - See https://en.wikipedia.org/wiki/LogSumExp#log-sum-exp_trick_for_log-domain_calculations
def logsoftmax(input: Tensor, dim: int) -> Tensor:
    """Compute the log of the softmax as a tensor.

    Args:
    ----
        input : input tensor
        dim : dimension to apply log-softmax

    Returns:
    -------
        Tensor : the log of the softmax tensor

    """
    maximium_i = max(input, dim=dim)
    output = input - maximium_i - ((input - maximium_i).exp().sum(dim=dim).log())
    return output


# - maxpool2d: Tiled max pooling 2D
def maxpool2d(input: Tensor, kernel: Tuple[int, int]) -> Tensor:
    """Tiled max pooling 2D.

    Args:
    ----
        input: batch x channel x height x width
        kernel: height x width of pooling

    Returns:
    -------
        Tensor : the pooled tensor

    """
    batch, channel, height, width = input.shape
    input, new_height, new_width = tile(input, kernel)
    output = (max(input, dim=4)).view(batch, channel, new_height, new_width)
    return output


# - dropout: Dropout positions based on random noise, include an argument to turn off
def dropout(input: Tensor, rate: float, ignore: bool = False) -> Tensor:
    """Dropout positions based on random noise, include an argument to turn off.

    Args:
    ----
        input : input tensor
        rate : probability [0, 1) of dropping out each position
        ignore : skip dropout, meaning doing nothing at all

    Returns:
    -------
        Tensor : the tensor with random positions dropped out

    """
    if ignore:
        output = input
    else:
        # Generate random boolean with the shape of input (0-1)
        random_bool = rand(input.shape) > rate
        output = input * random_bool
    return output
