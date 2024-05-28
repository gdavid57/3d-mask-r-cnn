"""
3D Mask R-CNN
Import of TensorFlow custom ops generalizing the TF
2D NonMaxSuppression and CropAndResize to the 3D case.
See custom_op folder for C source and compilation codes.

This 3D implementation was written by Gabriel David (PhD).

Licensed under the MIT License (see LICENSE for details)
"""

import tensorflow as tf
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops


#####################################################################
#  3D NonMaxSuppression and CropAndResize imports + grad bootstrap
#####################################################################

from crop_and_resize_3d import crop_and_resize_3d
from crop_and_resize_3d_grad_boxes import crop_and_resize_3d_grad_boxes
from crop_and_resize_3d_grad_image import crop_and_resize_3d_grad_image
from non_max_suppression_3d import non_max_suppression_3d


@ops.RegisterGradient("CropAndResize3D")
def _CropAndResize3DGrad(op, grad):
    """
    The derivatives for crop_and_resize.

    We back-propagate to the image only when the input image tensor has floating
    point dtype but we always back-propagate to the input boxes tensor.

    Args:
        op: The CropAndResize op.
        grad: The tensor representing the gradient w.r.t. the output.

    Returns:
        The gradients w.r.t. the input image, boxes, as well as the always-None
        gradients w.r.t. box_ind and crop_size.
    """
    image = op.inputs[0]
    if image.get_shape().is_fully_defined():
        image_shape = image.get_shape().as_list()
    else:
        image_shape = array_ops.shape(image)

    allowed_types = [dtypes.float16, dtypes.float32, dtypes.float64]
    if op.inputs[0].dtype in allowed_types:
        grad0 = crop_and_resize_3d_grad_image(grad, op.inputs[1], op.inputs[2], image_shape,
                                             T=op.get_attr("T"), method_name=op.get_attr("method_name"))
    else:
        grad0 = None

    # `grad0` is the gradient to the input image pixels and it
    # has been implemented for nearest neighbor and trilinear sampling
    # respectively. `grad1` is the gradient to the input crop boxes' coordinates.
    # When using nearest neighbor sampling, the gradient to crop boxes'
    # coordinates are not well defined. In practice, we still approximate
    # grad1 using the gradient derived from trilinear sampling.
    grad1 = crop_and_resize_3d_grad_boxes(grad, op.inputs[0], op.inputs[1], op.inputs[2])

    return [grad0, grad1, None, None]
