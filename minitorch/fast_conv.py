from typing import Tuple, TypeVar, Any

from numba import prange
from numba import njit as _njit

from .autodiff import Context
from .tensor import Tensor
from .tensor_data import (
    Shape,
    Strides,
    Storage,
    broadcast_index,
    index_to_position,
    to_index,
)
from .tensor_functions import Function

Fn = TypeVar("Fn")


def njit(fn: Fn, **kwargs: Any) -> Fn:
    """Wraps a function with Numba's just-in-time compilation with specific options."""
    return _njit(inline="always", **kwargs)(fn)  # type: ignore


# This code will JIT compile fast versions your tensor_data functions.
# If you get an error, read the docs for NUMBA as to what is allowed
# in these functions.
to_index = njit(to_index)
index_to_position = njit(index_to_position)
broadcast_index = njit(broadcast_index)


def _tensor_conv1d(
    out: Storage,
    out_shape: Shape,
    out_strides: Strides,
    out_size: int,
    input: Storage,
    input_shape: Shape,
    input_strides: Strides,
    weight: Storage,
    weight_shape: Shape,
    weight_strides: Strides,
    reverse: bool,
) -> None:
    """1D Convolution implementation.

    Given input tensor of

       `batch, in_channels, width`

    and weight tensor

       `out_channels, in_channels, k_width`

    Computes padded output of

       `batch, out_channels, width`

    `reverse` decides if weight is anchored left (False) or right.
    (See diagrams)

    Args:
    ----
        out (Storage): storage for `out` tensor.
        out_shape (Shape): shape for `out` tensor.
        out_strides (Strides): strides for `out` tensor.
        out_size (int): size of the `out` tensor.
        input (Storage): storage for `input` tensor.
        input_shape (Shape): shape for `input` tensor.
        input_strides (Strides): strides for `input` tensor.
        weight (Storage): storage for `input` tensor.
        weight_shape (Shape): shape for `input` tensor.
        weight_strides (Strides): strides for `input` tensor.
        reverse (bool): anchor weight at left or right

    """
    batch_, out_channels, out_width = out_shape
    batch, in_channels, width = input_shape
    out_channels_, in_channels_, kw = weight_shape

    assert (
        batch == batch_
        and in_channels == in_channels_
        and out_channels == out_channels_
    )
    s1 = input_strides
    s2 = weight_strides

    # TODO: Implement for Task 4.1.
    for curr_batch_idx in prange(batch_):
        for curr_out_channel_idx in prange(out_channels):
            for curr_out_width_idx in prange(out_width):
                # Compute the index of the current output element
                # out[curr_batch, curr_out_channel, curr_width]
                out_pos_ordinal = (
                    out_strides[0] * curr_batch_idx
                    + out_strides[1] * curr_out_channel_idx
                    + out_strides[2] * curr_out_width_idx
                )

                for curr_in_channel_idx in prange(in_channels):
                    for curr_kw in prange(kw):
                        # Determine the range of kernel positions based on the reverse flag
                        # `reverse` decides if weight is anchored left (False) or right.
                        if reverse:
                            curr_kw = kw - curr_kw - 1

                        # Compute input and weight position indices
                        # weight[curr_out_channel, curr_in_channel_idx, curr_kw]
                        weight_pos_ordinal = (
                            s2[0] * curr_out_channel_idx
                            + s2[1] * curr_in_channel_idx
                            + s2[2] * curr_kw
                        )
                        input_pos_ordinal = 0
                        width_offset_backward = curr_out_width_idx - curr_kw
                        width_offset_forward = curr_out_width_idx + curr_kw
                        if reverse and 0 <= width_offset_backward:
                            # input[curr_batch, curr_in_channel_idx , curr_width - curr_kw]
                            input_pos_ordinal = (
                                s1[0] * curr_batch_idx
                                + s1[1] * curr_in_channel_idx
                                + s1[2] * width_offset_backward
                            )
                        elif not reverse and width_offset_forward < width:
                            # input[curr_batch, curr_in_channel_idx , curr_width - curr_kw]
                            input_pos_ordinal = (
                                s1[0] * curr_batch_idx
                                + s1[1] * curr_in_channel_idx
                                + s1[2] * width_offset_forward
                            )
                        # Accumulate the final convolution result
                        out[out_pos_ordinal] += (
                            input[input_pos_ordinal] * weight[weight_pos_ordinal]
                        )


tensor_conv1d = njit(_tensor_conv1d, parallel=True)


class Conv1dFun(Function):
    @staticmethod
    def forward(ctx: Context, input: Tensor, weight: Tensor) -> Tensor:
        """Compute a 1D Convolution

        Args:
        ----
            ctx : Context
            input : batch x in_channel x h x w
            weight : out_channel x in_channel x kh x kw

        Returns:
        -------
            batch x out_channel x h x w

        """
        ctx.save_for_backward(input, weight)
        batch, in_channels, w = input.shape
        out_channels, in_channels2, kw = weight.shape
        assert in_channels == in_channels2

        # Run convolution
        output = input.zeros((batch, out_channels, w))
        tensor_conv1d(
            *output.tuple(), output.size, *input.tuple(), *weight.tuple(), False
        )
        return output

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, Tensor]:
        """Compute the gradient of a 1D Convolution with respect to the input and weight tensors during the backward pass.

        Args:
        ----
            ctx : Context - Stores the saved input and weight tensors from the forward pass.
            grad_output : Tensor - Gradient of the loss with respect to the output of the convolution (batch x out_channel x w).

        Returns:
        -------
            Tuple[Tensor, Tensor]
                - Gradient with respect to the input tensor (batch x in_channel x w).
                - Gradient with respect to the weight tensor (out_channel x in_channel x kw).

        """
        input, weight = ctx.saved_values
        batch, in_channels, w = input.shape
        out_channels, in_channels, kw = weight.shape
        grad_weight = grad_output.zeros((in_channels, out_channels, kw))
        new_input = input.permute(1, 0, 2)
        new_grad_output = grad_output.permute(1, 0, 2)
        tensor_conv1d(  # type: ignore
            *grad_weight.tuple(),
            grad_weight.size,
            *new_input.tuple(),
            *new_grad_output.tuple(),
            False,  # type: ignore
        )
        grad_weight = grad_weight.permute(1, 0, 2)

        grad_input = input.zeros((batch, in_channels, w))
        new_weight = weight.permute(1, 0, 2)
        tensor_conv1d(  # type: ignore
            *grad_input.tuple(),
            grad_input.size,  # type: ignore
            *grad_output.tuple(),
            *new_weight.tuple(),
            True,  # type: ignore
        )
        return grad_input, grad_weight


conv1d = Conv1dFun.apply


def _tensor_conv2d(
    out: Storage,
    out_shape: Shape,
    out_strides: Strides,
    out_size: int,
    input: Storage,
    input_shape: Shape,
    input_strides: Strides,
    weight: Storage,
    weight_shape: Shape,
    weight_strides: Strides,
    reverse: bool,
) -> None:
    """2D Convolution implementation.

    Given input tensor of

       `batch, in_channels, height, width`

    and weight tensor

       `out_channels, in_channels, k_height, k_width`

    Computes padded output of

       `batch, out_channels, height, width`

    `Reverse` decides if weight is anchored top-left (False) or bottom-right.
    (See diagrams)


    Args:
    ----
        out (Storage): storage for `out` tensor.
        out_shape (Shape): shape for `out` tensor.
        out_strides (Strides): strides for `out` tensor.
        out_size (int): size of the `out` tensor.
        input (Storage): storage for `input` tensor.
        input_shape (Shape): shape for `input` tensor.
        input_strides (Strides): strides for `input` tensor.
        weight (Storage): storage for `input` tensor.
        weight_shape (Shape): shape for `input` tensor.
        weight_strides (Strides): strides for `input` tensor.
        reverse (bool): anchor weight at top-left or bottom-right

    """
    batch_, out_channels, _, _ = out_shape
    batch, in_channels, height, width = input_shape
    out_channels_, in_channels_, kh, kw = weight_shape

    assert (
        batch == batch_
        and in_channels == in_channels_
        and out_channels == out_channels_
    )

    s1 = input_strides
    s2 = weight_strides
    # inners
    s10, s11, s12, s13 = s1[0], s1[1], s1[2], s1[3]
    s20, s21, s22, s23 = s2[0], s2[1], s2[2], s2[3]

    # TODO: Implement for Task 4.2.
    for curr_batch_idx in prange(batch_):
        for curr_out_channels_idx in prange(out_channels):
            # out_shape[2] == height_
            for curr_out_height_idx in prange(out_shape[2]):
                # out_shape[3] == width_
                for curr_out_width_idx in prange(out_shape[3]):
                    out_pos_ordinal = (
                        out_strides[0] * curr_batch_idx
                        + out_strides[1] * curr_out_channels_idx
                        + out_strides[2] * curr_out_height_idx
                        + out_strides[3] * curr_out_width_idx
                    )
                    for curr_in_channel_idx in prange(in_channels):
                        # handle and loop through for each kh and each kw
                        for curr_kh in prange(kh):
                            for curr_kw in prange(kw):
                                if reverse:
                                    height_offset_backward = (
                                        curr_out_height_idx - curr_kh
                                    )
                                    width_offset_backward = curr_out_width_idx - curr_kw
                                    # if reverse flagged and the offset is larger and equal to zero
                                    if (
                                        height_offset_backward >= 0
                                        and width_offset_backward >= 0
                                    ):
                                        input_inner_pos_ordinal = (
                                            s10 * curr_batch_idx
                                            + s11 * curr_in_channel_idx
                                            + s12 * height_offset_backward
                                            + s13 * width_offset_backward
                                        )
                                        weight_inner_pos_ordinal = (
                                            s20 * curr_out_channels_idx
                                            + s21 * curr_in_channel_idx
                                            + s22 * curr_kh
                                            + s23 * curr_kw
                                        )
                                        out[out_pos_ordinal] += (
                                            input[input_inner_pos_ordinal]
                                            * weight[weight_inner_pos_ordinal]
                                        )
                                    else:
                                        out[out_pos_ordinal] += 0
                                else:
                                    height_offset_forward = (
                                        curr_out_height_idx + curr_kh
                                    )
                                    width_offset_forward = curr_out_width_idx + curr_kw
                                    # if not reverse flagged and the offset is smaller than height or width
                                    if (
                                        height_offset_forward < height
                                        and width_offset_forward < width
                                    ):
                                        input_inner_pos_ordinal = (
                                            s10 * curr_batch_idx
                                            + s11 * curr_in_channel_idx
                                            + s12 * height_offset_forward
                                            + s13 * width_offset_forward
                                        )
                                        weight_inner_pos_ordinal = (
                                            s20 * curr_out_channels_idx
                                            + s21 * curr_in_channel_idx
                                            + s22 * curr_kh
                                            + s23 * curr_kw
                                        )
                                        out[out_pos_ordinal] += (
                                            input[input_inner_pos_ordinal]
                                            * weight[weight_inner_pos_ordinal]
                                        )
                                    else:
                                        out[out_pos_ordinal] += 0


tensor_conv2d = njit(_tensor_conv2d, parallel=True, fastmath=True)


class Conv2dFun(Function):
    @staticmethod
    def forward(ctx: Context, input: Tensor, weight: Tensor) -> Tensor:
        """Compute a 2D Convolution

        Args:
        ----
            ctx : Context
            input : batch x in_channel x h x w
            weight  : out_channel x in_channel x kh x kw

        Returns:
        -------
            (:class:`Tensor`) : batch x out_channel x h x w

        """
        ctx.save_for_backward(input, weight)
        batch, in_channels, h, w = input.shape
        out_channels, in_channels2, kh, kw = weight.shape
        assert in_channels == in_channels2
        output = input.zeros((batch, out_channels, h, w))
        tensor_conv2d(
            *output.tuple(), output.size, *input.tuple(), *weight.tuple(), False
        )
        return output

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, Tensor]:
        """Compute the gradient of a 2D Convolution with respect to the input and weight tensors during the backward pass.

        Args:
        ----
            ctx : Context - Stores the saved input and weight tensors from the forward pass.
            grad_output : Tensor - Gradient of the loss with respect to the output of the convolution (batch x out_channel x h x w).

        Returns:
        -------
            Tuple[Tensor, Tensor]
                - Gradient with respect to the input tensor (batch x in_channel x h x w).
                - Gradient with respect to the weight tensor (out_channel x in_channel x kh x kw).

        """
        input, weight = ctx.saved_values
        batch, in_channels, h, w = input.shape
        out_channels, in_channels, kh, kw = weight.shape

        grad_weight = grad_output.zeros((in_channels, out_channels, kh, kw))
        new_input = input.permute(1, 0, 2, 3)
        new_grad_output = grad_output.permute(1, 0, 2, 3)
        tensor_conv2d(  # type: ignore
            *grad_weight.tuple(),
            grad_weight.size,
            *new_input.tuple(),
            *new_grad_output.tuple(),
            False,  # type: ignore
        )
        grad_weight = grad_weight.permute(1, 0, 2, 3)

        grad_input = input.zeros((batch, in_channels, h, w))
        new_weight = weight.permute(1, 0, 2, 3)
        tensor_conv2d(  # type: ignore
            *grad_input.tuple(),
            grad_input.size,  # type: ignore
            *grad_output.tuple(),
            *new_weight.tuple(),
            True,  # type: ignore
        )
        return grad_input, grad_weight


conv2d = Conv2dFun.apply
