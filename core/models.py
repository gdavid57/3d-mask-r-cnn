import os
import re
import math
import platform
import multiprocessing
import numpy as np
import pandas as pd
from tqdm import tqdm
from skimage.io import imsave, imread
from sklearn.metrics import confusion_matrix

from core import utils

import tensorflow as tf
import keras
import keras.backend as K
import keras.layers as KL
import keras.engine as KE
import keras.models as KM

if platform.processor() == 'ppc64le':
    import core.custom_op.ppc64le_custom_op as custom_op
else:
    import core.custom_op.custom_op as custom_op

from core.utils import rpn_evaluation, head_evaluation, segbranch_evaluation, compute_ap, denorm_boxes, compute_overlaps
from core.data_generators import RPNGenerator, HeadGenerator, SegBranchGenerator, MrcnnGenerator, EmbryoDataset, EmbryoHeadDataset

# Requires TensorFlow 1.3+ and Keras 2.0.8+.
from distutils.version import LooseVersion

assert LooseVersion(tf.__version__) >= LooseVersion("2.1")
assert LooseVersion(keras.__version__) >= LooseVersion('2.3.1')


############################################################
#  Utility Functions
############################################################

class BatchNorm(KL.BatchNormalization):
    """Extends the Keras BatchNormalization class to allow a central place
    to make changes if needed.

    Batch normalization has a negative effect on training if batches are small
    so this layer is often frozen (via setting in Config class) and functions
    as linear layer.
    """

    def call(self, inputs, training=None):
        """
        Note about training values:
            None: Train BN layers. This is the normal mode
            False: Freeze BN layers. Good when batch size is small
            True: (don't use). Set layer in training mode even when making inferences
        """
        return super(self.__class__, self).call(inputs, training=training)


def compute_backbone_shapes(config, image_shape):
    """Computes the width and height of each stage of the backbone network.

    Returns:
        [N, (height, width)]. Where N is the number of stages
    """
    if callable(config.BACKBONE):
        return config.COMPUTE_BACKBONE_SHAPE(image_shape)

    # Currently supports ResNet only
    assert config.BACKBONE in ["resnet50", "resnet101"]
    return np.array(
        [[int(math.ceil(image_shape[0] / stride)),
          int(math.ceil(image_shape[1] / stride)),
          int(math.ceil(image_shape[2] / stride))]
         for stride in config.BACKBONE_STRIDES])


############################################################
#  Resnet Graph
############################################################

# Code adopted from:
# https://github.com/fchollet/deep-learning-models/blob/master/resnet50.py

def identity_block(input_tensor, kernel_size, filters, stage, block,
                   use_bias=True, train_bn=True):
    """The identity_block is the block that has no conv layer at shortcut
    # Arguments
        input_tensor: input tensor
        kernel_size: default 3, the kernel size of middle conv layer at main path
        filters: list of integers, the nb_filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
        use_bias: Boolean. To use or not use a bias in conv layers.
        train_bn: Boolean. Train or freeze Batch Norm layers
    """

    nb_filter1, nb_filter2, nb_filter3 = filters
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = KL.Conv3D(nb_filter1, (1, 1, 1), name=conv_name_base + '2a', use_bias=use_bias)(input_tensor)
    x = BatchNorm(name=bn_name_base + '2a')(x, training=train_bn)
    x = KL.Activation('relu')(x)

    x = KL.Conv3D(nb_filter2, (kernel_size, kernel_size, kernel_size), padding='same', 
                  name=conv_name_base + '2b', use_bias=use_bias)(x)
    x = BatchNorm(name=bn_name_base + '2b')(x, training=train_bn)
    x = KL.Activation('relu')(x)

    x = KL.Conv3D(nb_filter3, (1, 1, 1), name=conv_name_base + '2c', use_bias=use_bias)(x)
    x = BatchNorm(name=bn_name_base + '2c')(x, training=train_bn)

    x = KL.Add()([x, input_tensor])
    x = KL.Activation('relu', name='res' + str(stage) + block + '_out')(x)

    return x


def conv_block(input_tensor, kernel_size, filters, stage, block,
               strides=(2, 2, 2), use_bias=True, train_bn=True):
    """conv_block is the block that has a conv layer at shortcut
    # Arguments
        input_tensor: input tensor
        kernel_size: default 3, the kernel size of middle conv layer at main path
        filters: list of integers, the nb_filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
        use_bias: Boolean. To use or not use a bias in conv layers.
        train_bn: Boolean. Train or freeze Batch Norm layers
    Note that from stage 3, the first conv layer at main path is with subsample=(2,2)
    And the shortcut should have subsample=(2,2) as well
    """

    nb_filter1, nb_filter2, nb_filter3 = filters
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = KL.Conv3D(nb_filter1, (1, 1, 1), strides=strides, name=conv_name_base + '2a', 
                  use_bias=use_bias)(input_tensor)
    x = BatchNorm(name=bn_name_base + '2a')(x, training=train_bn)
    x = KL.Activation('relu')(x)

    x = KL.Conv3D(nb_filter2, (kernel_size, kernel_size, kernel_size), padding='same', 
                  name=conv_name_base + '2b', use_bias=use_bias)(x)
    x = BatchNorm(name=bn_name_base + '2b')(x, training=train_bn)
    x = KL.Activation('relu')(x)

    x = KL.Conv3D(nb_filter3, (1, 1, 1), name=conv_name_base + '2c', use_bias=use_bias)(x)
    x = BatchNorm(name=bn_name_base + '2c')(x, training=train_bn)

    shortcut = KL.Conv3D(nb_filter3, (1, 1, 1), strides=strides, name=conv_name_base + '1', 
                         use_bias=use_bias)(input_tensor)
    shortcut = BatchNorm(name=bn_name_base + '1')(shortcut, training=train_bn)

    x = KL.Add()([x, shortcut])
    x = KL.Activation('relu', name='res' + str(stage) + block + '_out')(x)

    return x


def resnet_graph(input_image, architecture, stage5=False, train_bn=True):
    """Build a ResNet graph.
        architecture: Can be resnet50 or resnet101
        stage5: Boolean. If False, stage5 of the network is not created
        train_bn: Boolean. Train or freeze Batch Norm layers
    """

    assert architecture in ["resnet50", "resnet101"]

    # Stage 1
    x = KL.ZeroPadding3D((3, 3, 3))(input_image)
    x = KL.Conv3D(64, (7, 7, 7), strides=(2, 2, 2), name='conv1', use_bias=True)(x)
    x = BatchNorm(name='bn_conv1')(x, training=train_bn)
    x = KL.Activation('relu')(x)
    C1 = x = KL.MaxPooling3D((3, 3, 3), strides=(2, 2, 2), padding="same")(x)

    # Stage 2
    x = conv_block(x, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1, 1), train_bn=train_bn)
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='b', train_bn=train_bn)
    C2 = x = identity_block(x, 3, [64, 64, 256], stage=2, block='c', train_bn=train_bn)

    # Stage 3
    x = conv_block(x, 3, [128, 128, 512], stage=3, block='a', train_bn=train_bn)
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='b', train_bn=train_bn)
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='c', train_bn=train_bn)
    C3 = x = identity_block(x, 3, [128, 128, 512], stage=3, block='d', train_bn=train_bn)

    # Stage 4
    x = conv_block(x, 3, [256, 256, 1024], stage=4, block='a', train_bn=train_bn)
    block_count = {"resnet50": 5, "resnet101": 22}[architecture]
    for i in range(block_count):
        x = identity_block(x, 3, [256, 256, 1024], stage=4, block=chr(98 + i), train_bn=train_bn)
    C4 = x

    # Stage 5
    if stage5:

        x = conv_block(x, 3, [512, 512, 2048], stage=5, block='a', train_bn=train_bn)
        x = identity_block(x, 3, [512, 512, 2048], stage=5, block='b', train_bn=train_bn)
        C5 = x = identity_block(x, 3, [512, 512, 2048], stage=5, block='c', train_bn=train_bn)

    else:

        C5 = None

    return [C1, C2, C3, C4, C5]


############################################################
#  Proposal Layer
############################################################

def apply_box_deltas_graph(boxes, deltas):
    """
    Applies the given deltas to the given boxes.

    Args:
        boxes: [N, (y1, x1, z1, y2, x2, z2)] boxes to update
        deltas: [N, (dy, dx, dz, log(dh), log(dw), log(dd))] refinements to apply

    Returns:
        result: the corrected boxes
    """

    # Convert to y, x, z, h, w, d
    height = boxes[:, 3] - boxes[:, 0]
    width = boxes[:, 4] - boxes[:, 1]
    depth = boxes[:, 5] - boxes[:, 2]
    center_y = boxes[:, 0] + 0.5 * height
    center_x = boxes[:, 1] + 0.5 * width
    center_z = boxes[:, 2] + 0.5 * depth

    # Apply deltas
    center_y += deltas[:, 0] * height
    center_x += deltas[:, 1] * width
    center_z += deltas[:, 2] * depth
    height *= tf.exp(deltas[:, 3])
    width *= tf.exp(deltas[:, 4])
    depth *= tf.exp(deltas[:, 5])

    # Convert back to y1, x1, z1, y2, x2, z2
    y1 = center_y - 0.5 * height
    x1 = center_x - 0.5 * width
    z1 = center_z - 0.5 * depth
    y2 = y1 + height
    x2 = x1 + width
    z2 = z1 + depth
    result = tf.stack([y1, x1, z1, y2, x2, z2], axis=1, name="apply_box_deltas_out")

    return result


def clip_boxes_graph(boxes, window):
    """
    Clip the anchor boxes within the normalized image space.

    Args:
        boxes: [N, (y1, x1, z1, y2, x2, z2)]
        window: [6] in the form y1, x1, z1, y2, x2, z2

    Returns:
        clipped: the clipped boxes
    """
    # Split
    wy1, wx1, wz1, wy2, wx2, wz2 = tf.split(window, 6)
    y1, x1, z1, y2, x2, z2 = tf.split(boxes, 6, axis=1)
    # Clip
    y1 = tf.maximum(tf.minimum(y1, wy2), wy1)
    x1 = tf.maximum(tf.minimum(x1, wx2), wx1)
    z1 = tf.maximum(tf.minimum(z1, wz2), wz1)
    y2 = tf.maximum(tf.minimum(y2, wy2), wy1)
    x2 = tf.maximum(tf.minimum(x2, wx2), wx1)
    z2 = tf.maximum(tf.minimum(z2, wz2), wz1)
    clipped = tf.concat([y1, x1, z1, y2, x2, z2], axis=1, name="clipped_boxes")
    clipped.set_shape((clipped.shape[0], 6))
    return clipped


class ProposalLayer(KE.Layer):
    """Receives anchor scores and selects a subset to pass as proposals
    to the second stage. Filtering is done based on anchor scores and
    non-max suppression to remove overlaps. It also applies bounding
    box refinement deltas to anchors.

    Inputs:
        rpn_probs: [batch, num_anchors, (bg prob, fg prob)]
        rpn_bbox: [batch, num_anchors, (dy, dx, dz, log(dh), log(dw), log(dd))]
        anchors: [batch, num_anchors, (y1, x1, z1, y2, x2, z2)] anchors in normalized coordinates

    Returns:
        Proposals in normalized coordinates [batch, rois, (y1, x1, z1, y2, x2, z2)]
    """

    def __init__(self, proposal_count, nms_threshold, pre_nms_limit, images_per_gpu, 
                 rpn_bbox_std_dev, **kwargs):
        super(ProposalLayer, self).__init__(**kwargs)

        self.proposal_count = proposal_count
        self.nms_threshold = nms_threshold
        self.pre_nms_limit = pre_nms_limit
        self.images_per_gpu = images_per_gpu
        self.rpn_bbox_std_dev = rpn_bbox_std_dev

    def call(self, inputs):

        # Box Scores. Use the foreground class confidence. [Batch, num_rois, 1]
        scores = inputs[0][:, :, 1]

        # Box deltas [batch, num_rois, 6]
        deltas = inputs[1]
        deltas = deltas * np.reshape(self.rpn_bbox_std_dev, [1, 1, 6])

        # Anchors
        anchors = inputs[2]

        # Improve performance by trimming to top anchors by score and doing the rest on the smaller subset.
        pre_nms_limit = tf.minimum(self.pre_nms_limit, tf.shape(anchors)[1])
        ix = tf.nn.top_k(scores, pre_nms_limit, sorted=True, name="top_anchors").indices

        scores = utils.batch_slice(
            [scores, ix], 
            lambda x, y: tf.gather(x, y), 
            self.images_per_gpu
        )

        deltas = utils.batch_slice(
            [deltas, ix], 
            lambda x, y: tf.gather(x, y), 
            self.images_per_gpu
        )

        pre_nms_anchors = utils.batch_slice(
            [anchors, ix], 
            lambda a, x: tf.gather(a, x),
            self.images_per_gpu,
            names=["pre_nms_anchors"]
        )

        # Apply deltas to anchors to get refined anchors.
        # [batch, N, (y1, x1, z1, y2, x2, z2)]
        boxes = utils.batch_slice(
            [pre_nms_anchors, deltas], 
            lambda x, y: apply_box_deltas_graph(x, y),
            self.images_per_gpu,
            names=["refined_anchors"]
        )

        # Clip to image boundaries. Since we're in normalized coordinates,
        # clip to 0..1 range. [batch, N, (y1, x1, z1, y2, x2, z2)]
        window = np.array([0, 0, 0, 1, 1, 1], dtype=np.float32)
        boxes = utils.batch_slice(
            boxes,
            lambda x: clip_boxes_graph(x, window),
            self.images_per_gpu,
            names=["refined_anchors_clipped"]
        )

        # Non-max suppression
        def nms(boxes, scores):

            indices = custom_op.non_max_suppression_3d(
                boxes, 
                scores, 
                self.proposal_count,
                self.nms_threshold, 
                name="rpn_non_max_suppression"
            )

            proposals = tf.gather(boxes, indices)

            # Pad if needed
            padding = tf.maximum(self.proposal_count - tf.shape(proposals)[0], 0)
            proposals = tf.pad(proposals, [(0, padding), (0, 0)])

            return proposals

        proposals = utils.batch_slice([boxes, scores], nms, self.images_per_gpu)

        return proposals

    def compute_output_shape(self, input_shape):

        return (None, self.proposal_count, 6)


############################################################
#  Region Proposal Network (RPN)
############################################################

def rpn_graph(feature_map, anchors_per_location, anchor_stride):
    """
    Builds the computation graph of Region Proposal Network.

    Args:
        feature_map: backbone features [batch, height, width, depth, channels]
        anchors_per_location: number of anchors per pixel in the feature map
        anchor_stride: Controls the density of anchors. Typically 1 (anchors for
                       every pixel in the feature map), or 2 (every other pixel).
        shared_layer_size: channel number of the RPN shared block

    Returns:
        rpn_class_logits: [batch, H * W * D * anchors_per_location, 2] Anchor classifier logits (before softmax)
        rpn_probs: [batch, H * W * D * anchors_per_location, 2] Anchor classifier probabilities.
        rpn_bbox: [batch, H * W * D * anchors_per_location, (dy, dx, dz, log(dh), log(dw), log(dd))] Deltas to be
                  applied to anchors.
    """

    # Shared convolutional base of the RPN
    shared = KL.Conv3D(512, (3, 3, 3), padding='same', activation='relu', strides=anchor_stride,
                       name='rpn_conv_shared')(feature_map)

    # Anchor Score. [batch, height, width, depth, anchors per location * 2].
    x = KL.Conv3D(2 * anchors_per_location, (1, 1, 1), padding='valid',
                  activation='linear', name='rpn_class_raw')(shared)

    # Reshape to [batch, anchors, 2]
    rpn_class_logits = KL.Lambda(lambda t: tf.reshape(t, [tf.shape(t)[0], -1, 2]))(x)

    # Softmax on last dimension of BG/FG.
    rpn_probs = KL.Activation( "softmax", name="rpn_class_xxx")(rpn_class_logits)

    # Bounding box refinement. [batch, H, W, D, anchors per location * depth]
    # where depth is [x, y, z, log(w), log(h), log(d)]
    x = KL.Conv3D(anchors_per_location * 6, (1, 1, 1), padding="valid",
                  activation='linear', name='rpn_bbox_pred')(shared)

    # Reshape to [batch, anchors, 6]
    rpn_bbox = KL.Lambda(lambda t: tf.reshape(t, [tf.shape(t)[0], -1, 6]))(x)

    return [rpn_class_logits, rpn_probs, rpn_bbox]


def build_rpn_model(anchor_stride, anchors_per_location, channel):
    """
        Builds a Keras model of the Region Proposal Network.
        It wraps the RPN graph so it can be used multiple times with shared
        weights.

        Args:
            anchors_per_location: number of anchors per pixel in the feature map
            anchor_stride: Controls the density of anchors. Typically 1 (anchors for
                           every pixel in the feature map), or 2 (every other pixel).
            channels: number of channels of a backbone feature map.
            shared_layer_size: channel number of the RPN shared block

        Returns:
            rpn_class_logits: [batch, H * W * D * anchors_per_location, 2] Anchor classifier logits (before softmax)
            rpn_probs: [batch, H * W * D * anchors_per_location, 2] Anchor classifier probabilities.
            rpn_bbox: [batch, H * W * D * anchors_per_location, (dy, dx, dz, log(dh), log(dw), log(dd))] Deltas to be
                        applied to anchors.
        """
    
    input_feature_map = KL.Input(shape=[None, None, None, channel], name="input_rpn_feature_map")

    outputs = rpn_graph(input_feature_map, anchors_per_location, anchor_stride)

    return KM.Model([input_feature_map], outputs, name="rpn_model")


############################################################
#  ROIAlign Layer
############################################################

def log2_graph(x):
    """Implementation of Log2. TF doesn't have a native implementation."""

    return tf.math.log(x) / tf.math.log(2.0)


class PyramidROIAlign(KE.Layer):
    """
    Implements ROI Pooling on multiple levels of the feature pyramid.

    Params:
    - pool_shape: [pool_height, pool_width, pool_depth] of the output pooled regions. Usually [7, 7, 7]

    Inputs:
    - boxes: [batch, num_boxes, (y1, x1, z1, y2, x2, z2)] in normalized
             coordinates. Possibly padded with zeros if not enough
             boxes to fill the array.
    - image_meta: [batch, (meta data)] Image details. See compose_image_meta()
    - feature_maps: List of feature maps from different levels of the pyramid.
                    Each is [batch, height, width, depth, channels]

    Output:
    Pooled regions in the shape: [batch, num_boxes, pool_height, pool_width, pool_depth, channels].
    The width, height and depth are those specific in the pool_shape in the layer
    constructor.
    """

    def __init__(self, pool_shape, **kwargs):
        super(PyramidROIAlign, self).__init__(**kwargs)
        self.pool_shape = tuple(pool_shape)

    def call(self, inputs):
        # Crop boxes [batch, num_boxes, (y1, x1, y2, x2)] in normalized coords
        boxes = inputs[0]

        # Image meta
        # Holds details about the image. See compose_image_meta()
        image_meta = inputs[1]

        # Feature Maps. List of feature maps from different level of the
        # feature pyramid. Each is [batch, height, width, channels]
        feature_maps = inputs[2:]

        # Assign each ROI to a level in the pyramid based on the ROI area.
        y1, x1, z1, y2, x2, z2 = tf.split(boxes, 6, axis=2)
        h = y2 - y1
        w = x2 - x1
        d = z2 - z1
        # Use shape of first image. Images in a batch must have the same size.
        image_shape = parse_image_meta_graph(image_meta)['image_shape'][0]
        # Equation 1 in the Feature Pyramid Networks paper. Account for
        # the fact that our coordinates are normalized here.
        # e.g. a 224x224 ROI (in pixels) maps to P4
        image_area = tf.cast(image_shape[0] * image_shape[1] * image_shape[2], tf.float32)
        roi_level = log2_graph(tf.pow(h * w * d, 1/3) / (224.0 / tf.pow(image_area, 1/3)))
        roi_level = tf.minimum(5, tf.maximum(
            2, 4 + tf.cast(tf.round(roi_level), tf.int32)))
        roi_level = tf.squeeze(roi_level, 2)

        # Loop through levels and apply ROI pooling to each. P2 to P5.
        pooled = []
        box_to_level = []
        for i, level in enumerate(range(2, 6)):
            ix = tf.where(tf.equal(roi_level, level))
            level_boxes = tf.gather_nd(boxes, ix)

            # Box indices for crop_and_resize.
            box_indices = tf.cast(ix[:, 0], tf.int32)

            # Keep track of which box is mapped to which level
            box_to_level.append(ix)

            # Stop gradient propogation to ROI proposals
            level_boxes = tf.stop_gradient(level_boxes)
            box_indices = tf.stop_gradient(box_indices)

            # Crop and Resize
            # From Mask R-CNN paper: "We sample four regular locations, so
            # that we can evaluate either max or average pooling. In fact,
            # interpolating only a single value at each bin center (without
            # pooling) is nearly as effective."
            #
            # Result: [batch * num_boxes, pool_height, pool_width, pool_depth, channels]
            pooled.append(custom_op.crop_and_resize_3d(feature_maps[i], level_boxes, box_indices, self.pool_shape))

        # Pack pooled features into one tensor
        pooled = tf.concat(pooled, axis=0)

        # Pack box_to_level mapping into one array and add another
        # column representing the order of pooled boxes
        box_to_level = tf.concat(box_to_level, axis=0)
        box_range = tf.expand_dims(tf.range(tf.shape(box_to_level)[0]), 1)
        box_to_level = tf.concat([tf.cast(box_to_level, tf.int32), box_range],
                                 axis=1)

        # Rearrange pooled features to match the order of the original boxes
        # Sort box_to_level by batch then box index
        # TF doesn't have a way to sort by two columns, so merge them and sort.
        sorting_tensor = box_to_level[:, 0] * 100000 + box_to_level[:, 1]
        ix = tf.nn.top_k(sorting_tensor, k=tf.shape(
            box_to_level)[0]).indices[::-1]
        ix = tf.gather(box_to_level[:, 2], ix)
        pooled = tf.gather(pooled, ix)

        # Re-add the batch dimension
        shape = tf.concat([tf.shape(boxes)[:2], tf.shape(pooled)[1:]], axis=0)
        pooled = tf.reshape(pooled, shape)

        return pooled

    def compute_output_shape(self, input_shape):

        return input_shape[0][:2] + self.pool_shape + (input_shape[2][-1],)


############################################################
#  Detection Target Layer
############################################################

def overlaps_graph(boxes1, boxes2):
    """
    Computes IoU overlaps between two sets of boxes.

    Args:
        boxes1, boxes2: [N, (y1, x1, z1, y2, x2, z2)].

    Returns:
        overlaps: matrix of overlap
    """

    # 1. Tile boxes2 and repeat boxes1. This allows us to compare
    # every boxes1 against every boxes2 without loops.
    # TF doesn't have an equivalent to np.repeat() so simulate it
    # using tf.tile() and tf.reshape.
    b1 = tf.reshape(tf.tile(tf.expand_dims(boxes1, 1), [1, 1, tf.shape(boxes2)[0]]), [-1, 6])
    b2 = tf.tile(boxes2, [tf.shape(boxes1)[0], 1])

    # 2. Compute intersections
    b1_y1, b1_x1, b1_z1, b1_y2, b1_x2, b1_z2 = tf.split(b1, 6, axis=1)
    b2_y1, b2_x1, b2_z1, b2_y2, b2_x2, b2_z2 = tf.split(b2, 6, axis=1)
    y1 = tf.maximum(b1_y1, b2_y1)
    x1 = tf.maximum(b1_x1, b2_x1)
    z1 = tf.maximum(b1_z1, b2_z1)
    y2 = tf.minimum(b1_y2, b2_y2)
    x2 = tf.minimum(b1_x2, b2_x2)
    z2 = tf.minimum(b1_z2, b2_z2)
    intersection = tf.maximum(x2 - x1, 0) * tf.maximum(y2 - y1, 0) * tf.maximum(z2 - z1, 0)

    # 3. Compute unions
    b1_area = (b1_y2 - b1_y1) * (b1_x2 - b1_x1) * (b1_z2 - b1_z1)
    b2_area = (b2_y2 - b2_y1) * (b2_x2 - b2_x1) * (b2_z2 - b2_z1)
    union = b1_area + b2_area - intersection

    # 4. Compute IoU and reshape to [boxes1, boxes2]
    iou = intersection / union
    overlaps = tf.reshape(iou, [tf.shape(boxes1)[0], tf.shape(boxes2)[0]])

    return overlaps


def detection_targets_graph(proposals, gt_class_ids, gt_boxes, gt_masks, train_rois_per_image,
                            roi_positive_ratio, bbox_std_dev, use_mini_mask, mask_shape):
    """
    Generates detection targets for one image. Subsamples proposals and
    generates target class IDs, bounding box deltas, and masks for each.

    Inputs:
        proposals: [POST_NMS_ROIS_TRAINING, (y1, x1, z1, y2, x2, z2)] in normalized coordinates. Might
                   be zero padded if there are not enough proposals.
        gt_class_ids: [MAX_GT_INSTANCES] int class IDs
        gt_boxes: [MAX_GT_INSTANCES, (y1, x1, z1, y2, x2, z2)] in normalized coordinates.
        gt_masks: [height, width, depth, MAX_GT_INSTANCES] of boolean type.

    Returns: Target ROIs and corresponding class IDs, bounding box shifts,
    and masks.
        rois: [TRAIN_ROIS_PER_IMAGE, (y1, x1, z1, y2, x2, z2)] in normalized coordinates
        class_ids: [TRAIN_ROIS_PER_IMAGE]. Integer class IDs. Zero padded.
        deltas: [TRAIN_ROIS_PER_IMAGE, (dy, dx, dz, log(dh), log(dw), log(dd) )]
        masks: [TRAIN_ROIS_PER_IMAGE, height, width, depth]. Masks cropped to bbox
               boundaries and resized to neural network output size.

    Note: Returned arrays might be zero padded if not enough target ROIs.
    """

    # Assertions
    asserts = [
        tf.Assert(tf.greater(tf.shape(proposals)[0], 0), [proposals], name="roi_assertion")
    ]
    with tf.control_dependencies(asserts):
        proposals = tf.identity(proposals)

    # Remove zero padding
    proposals, _ = trim_zeros_graph(proposals, name="trim_proposals")
    gt_boxes, non_zeros = trim_zeros_graph(gt_boxes, name="trim_gt_boxes")
    gt_class_ids = tf.boolean_mask(gt_class_ids, non_zeros,
                                   name="trim_gt_class_ids")
    gt_masks = tf.gather(gt_masks, tf.where(non_zeros)[:, 0], axis=3,
                         name="trim_gt_masks")

    # Compute overlaps matrix [proposals, gt_boxes]
    overlaps = overlaps_graph(proposals, gt_boxes)

    # Determine positive and negative ROIs
    roi_iou_max = tf.reduce_max(overlaps, axis=1)

    # 1. Positive ROIs are those with >= 0.5 IoU with a GT box
    positive_roi_bool = (roi_iou_max >= 0.5)
    positive_indices = tf.where(positive_roi_bool)[:, 0]

    # 2. Negative ROIs are those with < 0.5 with every GT box. Skip crowds.
    negative_indices = tf.where(roi_iou_max < 0.5)[:, 0]

    # Subsample ROIs.
    # Positive ROIs
    positive_count = int(train_rois_per_image * roi_positive_ratio)

    positive_indices = tf.random.shuffle(positive_indices)[:positive_count]
    positive_count = tf.shape(positive_indices)[0]

    # Negative ROIs. Add enough to maintain positive:negative ratio.
    # r = 1.0 / config.ROI_POSITIVE_RATIO
    # negative_count = tf.cast(r * tf.cast(positive_count, tf.float32), tf.int32) - positive_count
    negative_count = train_rois_per_image - positive_count
    negative_indices = tf.random.shuffle(negative_indices)[:negative_count]

    # Gather selected ROIs
    positive_rois = tf.gather(proposals, positive_indices)
    negative_rois = tf.gather(proposals, negative_indices)

    # Assign positive ROIs to GT boxes.
    positive_overlaps = tf.gather(overlaps, positive_indices)
    negative_overlaps = tf.gather(overlaps, negative_indices)
    roi_gt_box_pos_assignment = tf.argmax(positive_overlaps, axis=1)
    roi_gt_box_neg_assignment = tf.argmax(negative_overlaps, axis=1)

    roi_gt_class_ids = tf.gather(gt_class_ids, roi_gt_box_pos_assignment)

    roi_pos_gt_boxes = tf.gather(gt_boxes, roi_gt_box_pos_assignment)
    roi_neg_gt_boxes = tf.gather(gt_boxes, roi_gt_box_neg_assignment)

    # Compute bbox refinement for positive ROIs
    deltas = utils.box_refinement_graph(positive_rois, roi_pos_gt_boxes)
    deltas /= bbox_std_dev

    # Assign positive ROIs to GT masks
    # Permute masks to [N, height, width, 1]
    transposed_masks = tf.expand_dims(tf.transpose(gt_masks, [3, 0, 1, 2]), -1)

    # Pick the right mask for each ROI
    roi_masks = tf.gather(transposed_masks, roi_gt_box_pos_assignment)

    # Compute mask targets
    boxes = positive_rois

    if use_mini_mask:

        # Transform ROI coordinates from normalized image space
        # to normalized mini-mask space.
        # Transform ROI coordinates from normalized image space
        # to normalized mini-mask space.
        y1, x1, z1, y2, x2, z2 = tf.split(positive_rois, 6, axis=1)
        gt_y1, gt_x1, gt_z1, gt_y2, gt_x2, gt_z2 = tf.split(roi_pos_gt_boxes, 6, axis=1)
        gt_h = gt_y2 - gt_y1
        gt_w = gt_x2 - gt_x1
        gt_d = gt_z2 - gt_z1
        y1 = (y1 - gt_y1) / gt_h
        x1 = (x1 - gt_x1) / gt_w
        z1 = (z1 - gt_z1) / gt_d
        y2 = (y2 - gt_y1) / gt_h
        x2 = (x2 - gt_x1) / gt_w
        z2 = (z2 - gt_z1) / gt_d
        boxes = tf.concat([y1, x1, z1, y2, x2, z2], 1)

    box_ids = tf.range(0, tf.shape(roi_masks)[0])
    masks = custom_op.crop_and_resize_3d(tf.cast(roi_masks, tf.float32), boxes, box_ids, mask_shape)
    
    # Remove the extra dimension from masks.
    masks = tf.squeeze(masks, axis=4)

    # Threshold mask pixels at 0.5 to have GT masks be 0 or 1 to use with
    # binary cross entropy loss.
    masks = tf.round(masks)

    # Append negative ROIs and pad bbox deltas and masks that
    # are not used for negative ROIs with zeros.
    rois = tf.concat([positive_rois, negative_rois], axis=0)
    roi_gt_boxes = tf.concat([roi_pos_gt_boxes, roi_neg_gt_boxes], axis=0)
    N = tf.shape(negative_rois)[0]
    P = tf.maximum(train_rois_per_image - tf.shape(rois)[0], 0)
    rois = tf.pad(rois, [(0, P), (0, 0)])
    roi_gt_class_ids = tf.pad(roi_gt_class_ids, [(0, N + P)])
    deltas = tf.pad(deltas, [(0, N + P), (0, 0)])
    masks = tf.pad(masks, [[0, N + P], (0, 0), (0, 0), (0, 0)])

    return rois, roi_gt_boxes, roi_gt_class_ids, deltas, masks


class DetectionTargetLayer(KE.Layer):
    """
    Subsamples proposals and generates target box refinement, class_ids,
    and masks for each.

    Inputs:
        proposals: [batch, N, (y1, x1, z1, y2, x2, z2)] in normalized coordinates. Might
                   be zero padded if there are not enough proposals.
        gt_class_ids: [batch, MAX_GT_INSTANCES] Integer class IDs.
        gt_boxes: [batch, MAX_GT_INSTANCES, (y1, x1, z1, y2, x2, z2)] in normalized
                  coordinates.
        gt_masks: [batch, height, width, depth, MAX_GT_INSTANCES] of boolean type

    Returns: Target ROIs and corresponding class IDs, bounding box shifts,
    and masks.
        rois: [batch, TRAIN_ROIS_PER_IMAGE, (y1, x1, z1, y2, x2, z2)] in normalized
              coordinates
        target_class_ids: [batch, TRAIN_ROIS_PER_IMAGE]. Integer class IDs.
        target_deltas: [batch, TRAIN_ROIS_PER_IMAGE, (dy, dx, dz, log(dh), log(dw), log(dd)]
        target_mask: [batch, TRAIN_ROIS_PER_IMAGE, height, width, depth]
                     Masks cropped to bbox boundaries and resized to neural
                     network output size.

    Note: Returned arrays might be zero padded if not enough target ROIs.
    """

    def __init__(self, train_rois_per_image, roi_positive_ratio, bbox_std_dev, 
                 use_mini_mask, mask_shape, images_per_gpu, **kwargs):
        super(DetectionTargetLayer, self).__init__(**kwargs)
        self.train_rois_per_image = train_rois_per_image
        self.roi_positive_ratio = roi_positive_ratio
        self.bbox_std_dev = bbox_std_dev
        self.use_mini_mask = use_mini_mask
        self.mask_shape = mask_shape
        self.images_per_gpu = images_per_gpu

    def call(self, inputs):
        proposals = inputs[0]
        gt_class_ids = inputs[1]
        gt_boxes = inputs[2]
        gt_masks = inputs[3]

        # Slice the batch and run a graph for each slice
        names = ["rois", "target_gt_boxes", "target_class_ids", "target_bbox", "target_mask"]

        outputs = utils.batch_slice(
            [proposals, gt_class_ids, gt_boxes, gt_masks],
            lambda w, x, y, z: detection_targets_graph(
                w, x, y, z, 
                self.train_rois_per_image,
                self.roi_positive_ratio,
                self.bbox_std_dev,
                self.use_mini_mask,
                self.mask_shape
            ),
            self.images_per_gpu, 
            names=names
        )

        return outputs

    def compute_output_shape(self, input_shape):
        return [
            (None, self.train_rois_per_image, 6),  # rois
            (None, self.train_rois_per_image, 6),  # gt_boxes
            (None, self.train_rois_per_image),  # class_ids
            (None, self.train_rois_per_image, 6),  # deltas
            (None, self.train_rois_per_image, *self.mask_shape)  # masks
        ]

    def compute_mask(self, inputs, mask=None):
        return [None, None, None, None, None]


############################################################
#  Feature Pyramid Network Heads
############################################################

def fpn_classifier_graph(y, pool_size, num_classes, fc_layers_size, train_bn=True):
    """
    Builds the computation graph of the feature pyramid network classifier
    and regressor heads.

    rois: [batch, num_rois, (y1, x1, z1, y2, x2, z2)] Proposal boxes in normalized
          coordinates.
    feature_maps: List of feature maps from different layers of the pyramid,
                  [P2, P3, P4, P5]. Each has a different resolution.
    image_meta: [batch, (meta data)] Image details. See compose_image_meta()
    pool_size: The width of the square feature map generated from ROI Pooling.
    num_classes: number of classes, which determines the depth of the results
    train_bn: Boolean. Train or freeze Batch Norm layers
    fc_layers_size: Size of the 2 FC layers

    Returns:
        logits: [batch, num_rois, NUM_CLASSES] classifier logits (before softmax)
        probs: [batch, num_rois, NUM_CLASSES] classifier probabilities
        bbox_deltas: [batch, num_rois, NUM_CLASSES, (dy, dx, dz, log(dh), log(dw), log(dd))] Deltas to apply to
                     proposal boxes
    """

    s = K.int_shape(y)

    x = KL.TimeDistributed(KL.Conv3D(fc_layers_size, (pool_size, pool_size, pool_size), padding="valid"),
                           name="mrcnn_class_conv1")(y)
    x = KL.TimeDistributed(BatchNorm(), name='mrcnn_class_bn1')(x, training=train_bn)
    x = KL.Activation('relu')(x)

    x = KL.TimeDistributed(KL.Conv3D(fc_layers_size, (1, 1, 1)), name="mrcnn_class_conv2")(x)
    x = KL.TimeDistributed(BatchNorm(), name='mrcnn_class_bn2')(x, training=train_bn)
    x = KL.Activation('relu')(x)

    shared = KL.Reshape((s[1], fc_layers_size), name="pool_reshape")(x)

    # Classifier head
    mrcnn_class_logits = KL.TimeDistributed(KL.Dense(num_classes), name='mrcnn_class_logits')(shared)
    mrcnn_probs = KL.TimeDistributed(KL.Activation("softmax"), name="mrcnn_class")(mrcnn_class_logits)

    # BBox head
    x = KL.TimeDistributed(KL.Dense(num_classes * 6, activation='linear'), name='mrcnn_bbox_fc')(shared)

    # Reshape to [batch, num_rois, NUM_CLASSES, (dy, dx, dz, log(dh), log(dw), log(dd))]
    mrcnn_bbox = KL.Reshape((s[1], num_classes, 6), name="mrcnn_bbox")(x)

    return mrcnn_class_logits, mrcnn_probs, mrcnn_bbox


def build_fpn_mask_graph(y, num_classes, conv_channel, train_bn=True):
    """
    Builds the computation graph of the mask head of Feature Pyramid Network.

    rois: [batch, num_rois, (y1, x1, y2, x2)] Proposal boxes in normalized
          coordinates.
    feature_maps: List of feature maps from different layers of the pyramid,
                  [P2, P3, P4, P5]. Each has a different resolution.
    image_meta: [batch, (meta data)] Image details. See compose_image_meta()
    pool_size: The width of the square feature map generated from ROI Pooling.
    num_classes: number of classes, which determines the depth of the results
    train_bn: Boolean. Train or freeze Batch Norm layers

    Returns: Masks [batch, num_rois, MASK_POOL_SIZE, MASK_POOL_SIZE, MASK_POOL_SIZE, NUM_CLASSES]
    """

    # Conv layers
    x = KL.TimeDistributed(KL.Conv3D(conv_channel, (3, 3, 3), padding="same"), name="mrcnn_mask_conv1")(y)
    x = KL.TimeDistributed(BatchNorm(), name='mrcnn_mask_bn1')(x, training=train_bn)
    x = KL.Activation('relu')(x)

    x = KL.TimeDistributed(KL.Conv3D(conv_channel, (3, 3, 3), padding="same"), name="mrcnn_mask_conv2")(x)
    x = KL.TimeDistributed(BatchNorm(), name='mrcnn_mask_bn2')(x, training=train_bn)
    x = KL.Activation('relu')(x)

    x = KL.TimeDistributed(KL.Conv3D(conv_channel, (3, 3, 3), padding="same"), name="mrcnn_mask_conv3")(x)
    x = KL.TimeDistributed(BatchNorm(), name='mrcnn_mask_bn3')(x, training=train_bn)
    x = KL.Activation('relu')(x)

    x = KL.TimeDistributed(KL.Conv3D(conv_channel, (3, 3, 3), padding="same"), name="mrcnn_mask_conv4")(x)
    x = KL.TimeDistributed(BatchNorm(), name='mrcnn_mask_bn4')(x, training=train_bn)
    x = KL.Activation('relu')(x)

    x = KL.TimeDistributed(KL.Conv3DTranspose(conv_channel, (2, 2, 2), strides=2, activation="relu"), name="mrcnn_mask_deconv")(x)
    x = KL.TimeDistributed(KL.Conv3D(num_classes, (1, 1, 1), strides=1, activation="sigmoid"), name="mrcnn_mask")(x)

    return x


def fpn_classifier_graph_with_RoiAlign(rois, feature_maps, image_meta,
                         pool_size, num_classes, fc_layers_size, train_bn=True):
    """
    Builds the computation graph of the feature pyramid network classifier
    and regressor heads.

    rois: [batch, num_rois, (y1, x1, z1, y2, x2, z2)] Proposal boxes in normalized
          coordinates.
    feature_maps: List of feature maps from different layers of the pyramid,
                  [P2, P3, P4, P5]. Each has a different resolution.
    image_meta: [batch, (meta data)] Image details. See compose_image_meta()
    pool_size: The width of the square feature map generated from ROI Pooling.
    num_classes: number of classes, which determines the depth of the results
    train_bn: Boolean. Train or freeze Batch Norm layers
    fc_layers_size: Size of the 2 FC layers

    Returns:
        logits: [batch, num_rois, NUM_CLASSES] classifier logits (before softmax)
        probs: [batch, num_rois, NUM_CLASSES] classifier probabilities
        bbox_deltas: [batch, num_rois, NUM_CLASSES, (dy, dx, dz, log(dh), log(dw) log(dd))] Deltas to apply to
                     proposal boxes
    """
    
    s = K.int_shape(rois)

    # ROI Pooling
    # Shape: [batch, num_rois, POOL_SIZE, POOL_SIZE, POOL_SIZE, channels]
    x = PyramidROIAlign([pool_size, pool_size, pool_size], name="roi_align_classifier")([rois, image_meta] + feature_maps)
    
    # Two 1024 FC layers (implemented with Conv2D for consistency)
    x = KL.TimeDistributed(KL.Conv3D(fc_layers_size, (pool_size, pool_size, pool_size), padding="valid"),
                           name="mrcnn_class_conv1")(x)
    x = KL.TimeDistributed(BatchNorm(), name='mrcnn_class_bn1')(x, training=train_bn)
    x = KL.Activation('relu')(x)

    x = KL.TimeDistributed(KL.Conv3D(fc_layers_size, (1, 1, 1)), name="mrcnn_class_conv2")(x)
    x = KL.TimeDistributed(BatchNorm(), name='mrcnn_class_bn2')(x, training=train_bn)
    x = KL.Activation('relu')(x)

    shared = KL.Reshape((s[1], fc_layers_size), name="pool_reshape")(x)

    # Classifier head
    mrcnn_class_logits = KL.TimeDistributed(KL.Dense(num_classes), name='mrcnn_class_logits')(shared)
    mrcnn_probs = KL.TimeDistributed(KL.Activation("softmax"), name="mrcnn_class")(mrcnn_class_logits)

    # BBox head
    x = KL.TimeDistributed(KL.Dense(num_classes * 6, activation='linear'), name='mrcnn_bbox_fc')(shared)

    # Reshape to [batch, num_rois, NUM_CLASSES, (dy, dx, dz, log(dh), log(dw), log(dd))]
    mrcnn_bbox = KL.Reshape((s[1], num_classes, 6), name="mrcnn_bbox")(x)

    return mrcnn_class_logits, mrcnn_probs, mrcnn_bbox


def build_fpn_mask_graph_with_RoiAlign(rois, feature_maps, image_meta,
                         pool_size, num_classes, conv_channel, train_bn=True):
    """
    Builds the computation graph of the mask head of Feature Pyramid Network.

    rois: [batch, num_rois, (y1, x1, z1, y2, x2, z2)] Proposal boxes in normalized
          coordinates.
    feature_maps: List of feature maps from different layers of the pyramid,
                  [P2, P3, P4, P5]. Each has a different resolution.
    image_meta: [batch, (meta data)] Image details. See compose_image_meta()
    pool_size: The width of the square feature map generated from ROI Pooling.
    num_classes: number of classes, which determines the depth of the results
    train_bn: Boolean. Train or freeze Batch Norm layers

    Returns: Masks [batch, num_rois, MASK_POOL_SIZE, MASK_POOL_SIZE, MASK_POOL_SIZE, NUM_CLASSES]
    """

    # ROI Pooling
    # Shape: [batch, num_rois, MASK_POOL_SIZE, MASK_POOL_SIZE, channels]
    x = PyramidROIAlign([pool_size, pool_size, pool_size], name="roi_align_mask")([rois, image_meta] + feature_maps)

    # Conv layers
    x = KL.TimeDistributed(KL.Conv3D(conv_channel, (3, 3, 3), padding="same"), name="mrcnn_mask_conv1")(x)
    x = KL.TimeDistributed(BatchNorm(), name='mrcnn_mask_bn1')(x, training=train_bn)
    x = KL.Activation('relu')(x)

    x = KL.TimeDistributed(KL.Conv3D(conv_channel, (3, 3, 3), padding="same"), name="mrcnn_mask_conv2")(x)
    x = KL.TimeDistributed(BatchNorm(), name='mrcnn_mask_bn2')(x, training=train_bn)
    x = KL.Activation('relu')(x)

    x = KL.TimeDistributed(KL.Conv3D(conv_channel, (3, 3, 3), padding="same"), name="mrcnn_mask_conv3")(x)
    x = KL.TimeDistributed(BatchNorm(), name='mrcnn_mask_bn3')(x, training=train_bn)
    x = KL.Activation('relu')(x)

    x = KL.TimeDistributed(KL.Conv3D(conv_channel, (3, 3, 3), padding="same"), name="mrcnn_mask_conv4")(x)
    x = KL.TimeDistributed(BatchNorm(), name='mrcnn_mask_bn4')(x, training=train_bn)
    x = KL.Activation('relu')(x)

    x = KL.TimeDistributed(KL.Conv3DTranspose(conv_channel, (2, 2, 2), strides=2, activation="relu"), name="mrcnn_mask_deconv")(x)
    x = KL.TimeDistributed(KL.Conv3D(num_classes, (1, 1, 1), strides=1, activation="sigmoid"), name="mrcnn_mask")(x)

    return x


############################################################
#  Detection Layer
############################################################

def refine_detections_graph(rois, probs, deltas, window, bbox_std_dev, 
                            detection_min_confidence, detection_max_instances,
                            detection_nms_threshold):
    """
    Refine classified proposals and filter overlaps and return final
    detections.

    Inputs:
        rois: [N, (y1, x1, z1, y2, x2, z2)] in normalized coordinates
        probs: [N, num_classes]. Class probabilities.
        deltas: [N, num_classes, (dy, dx, dz, log(dh), log(dw), log(dd) )]. Class-specific
                bounding box deltas.
        window: (y1, x1, z1, y2, x2, z2) in normalized coordinates. The part of the image
            that contains the image excluding the padding.

    Returns detections shaped: [num_detections, (y1, x1, z1, y2, x2, z2, class_id, score)] where
        coordinates are normalized.
    """

    # Class IDs per ROI
    class_ids = tf.argmax(probs, axis=1, output_type=tf.int32)

    # Class probability of the top class of each ROI
    indices = tf.stack([tf.range(probs.shape[0]), class_ids], axis=1)
    class_scores = tf.gather_nd(probs, indices)

    # Class-specific bounding box deltas
    deltas_specific = tf.gather_nd(deltas, indices)

    # Apply bounding box deltas
    # Shape: [boxes, (y1, x1, z1, y2, x2, z2)] in normalized coordinates
    refined_rois = apply_box_deltas_graph(rois, deltas_specific * bbox_std_dev)
    # Clip boxes to image window
    refined_rois = clip_boxes_graph(refined_rois, window)

    # Filter out background boxes
    keep = tf.where(class_ids > 0)[:, 0]

    # Filter out low confidence boxes
    if detection_min_confidence:
        conf_keep = tf.where(class_scores >= detection_min_confidence)[:, 0]
        keep = tf.sets.intersection(tf.expand_dims(keep, 0), tf.expand_dims(conf_keep, 0))
        keep = tf.sparse.to_dense(keep)[0]

    # Apply per-class NMS
    # 1. Prepare variables
    pre_nms_class_ids = tf.gather(class_ids, keep)
    pre_nms_scores = tf.gather(class_scores, keep)
    pre_nms_rois = tf.gather(refined_rois, keep)
    unique_pre_nms_class_ids = tf.unique(pre_nms_class_ids)[0]

    def nms_keep_map(class_id):
        """Apply Non-Maximum Suppression on ROIs of the given class."""

        # Indices of ROIs of the given class
        ixs = tf.where(tf.equal(pre_nms_class_ids, class_id))[:, 0]

        # Apply NMS
        class_keep = custom_op.non_max_suppression_3d(
            tf.gather(pre_nms_rois, ixs),
            tf.gather(pre_nms_scores, ixs),
            max_output_size=detection_max_instances,
            iou_threshold=detection_nms_threshold
        )

        # Map indices
        class_keep = tf.gather(keep, tf.gather(ixs, class_keep))

        # Pad with -1 so returned tensors have the same shape
        gap = detection_max_instances - tf.shape(class_keep)[0]
        class_keep = tf.pad(class_keep, [(0, gap)], mode='CONSTANT', constant_values=-1)

        # Set shape so map_fn() can infer result shape
        class_keep.set_shape([detection_max_instances])

        return class_keep

    # 2. Map over class IDs
    nms_keep = tf.map_fn(nms_keep_map, unique_pre_nms_class_ids, dtype=tf.int64)

    # 3. Merge results into one list, and remove -1 padding
    nms_keep = tf.reshape(nms_keep, [-1])
    nms_keep = tf.gather(nms_keep, tf.where(nms_keep > -1)[:, 0])

    # 4. Compute intersection between keep and nms_keep
    keep = tf.sets.intersection(tf.expand_dims(keep, 0), tf.expand_dims(nms_keep, 0))
    keep = tf.sparse.to_dense(keep)[0]

    # Keep top detections
    roi_count = detection_max_instances
    class_scores_keep = tf.gather(class_scores, keep)
    num_keep = tf.minimum(tf.shape(class_scores_keep)[0], roi_count)
    top_ids = tf.nn.top_k(class_scores_keep, k=num_keep, sorted=True)[1]
    keep = tf.gather(keep, top_ids)

    # Arrange output as [N, (y1, x1, y2, x2, class_id, score)]
    # Coordinates are normalized.
    detections = tf.concat(
        [
        tf.gather(refined_rois, keep),
        tf.cast(tf.gather(class_ids, keep), tf.float32)[..., tf.newaxis],
        tf.gather(class_scores, keep)[..., tf.newaxis]
    ], axis=1)

    # Pad with zeros if detections < DETECTION_MAX_INSTANCES
    gap = detection_max_instances - tf.shape(detections)[0]
    detections = tf.pad(detections, [(0, gap), (0, 0)], "CONSTANT")

    return detections


class DetectionLayer(KE.Layer):
    """
    Takes classified proposal boxes and their bounding box deltas and
    returns the final detection boxes.

    Returns:
    [batch, num_detections, (y1, x1, z1, y2, x2, z2, class_id, class_score)] where
    coordinates are normalized.
    """

    def __init__(self, bbox_std_dev, detection_min_confidence, 
                 detection_max_instances, detection_nms_threshold, 
                 images_per_gpu, batch_size, **kwargs):
        super(DetectionLayer, self).__init__(**kwargs)
        self.bbox_std_dev = bbox_std_dev
        self.detection_min_confidence = detection_min_confidence
        self.detection_max_instances = detection_max_instances
        self.detection_nms_threshold = detection_nms_threshold
        self.images_per_gpu = images_per_gpu
        self.batch_size = batch_size

    def call(self, inputs):
        rois = inputs[0]
        mrcnn_class = inputs[1]
        mrcnn_bbox = inputs[2]
        image_meta = inputs[3]

        # Get windows of images in normalized coordinates. Windows are the area
        # in the image that excludes the padding.
        # Use the shape of the first image in the batch to normalize the window
        # because we know that all images get resized to the same size.
        m = parse_image_meta_graph(image_meta)
        image_shape = m['image_shape'][0]
        window = norm_boxes_graph(m['window'], image_shape[:3])

        # Run detection refinement graph on each item in the batch
        detections_batch = utils.batch_slice(
            [rois, mrcnn_class, mrcnn_bbox, window],
            lambda x, y, w, z: refine_detections_graph(
                x, 
                y, 
                w, 
                z, 
                self.bbox_std_dev,
                self.detection_min_confidence,
                self.detection_max_instances,
                self.detection_nms_threshold
            ),
            self.images_per_gpu
        )

        # Reshape output
        # [batch, num_detections, (y1, x1, z1, y2, x2, z2, class_id, class_scores)] in
        # normalized coordinates
        detections_batch = tf.reshape(detections_batch, [self.batch_size, self.detection_max_instances, 8])

        return detections_batch

    def compute_output_shape(self, input_shape):

        return (None, self.detection_max_instances, 8)


############################################################
#  Loss Functions
############################################################

def smooth_l1_loss(y_true, y_pred):
    """
    Implements Smooth-L1 loss.
    y_true and y_pred are typically: [N, 4], but could be any shape.
    """

    diff = K.abs(y_true - y_pred)
    less_than_one = K.cast(K.less(diff, 1.0), "float32")
    loss = (less_than_one * 0.5 * diff ** 2) + (1 - less_than_one) * (diff - 0.5)

    return loss


def rpn_class_loss_graph(rpn_match, rpn_class_logits):
    """
    RPN anchor classifier loss.

    rpn_match: [batch, anchors, 1]. Anchor match type. 1=positive,
               -1=negative, 0=neutral anchor.
    rpn_class_logits: [batch, anchors, 2]. RPN classifier logits for BG/FG.
    """

    # Squeeze last dim to simplify
    rpn_match = tf.squeeze(rpn_match, -1)

    # Get anchor classes. Convert the -1/+1 match to 0/1 values.
    anchor_class = K.cast(K.equal(rpn_match, 1), tf.int32)

    # Positive and Negative anchors contribute to the loss,
    # but neutral anchors (match value = 0) don't.
    indices = tf.where(K.not_equal(rpn_match, 0))

    # Pick rows that contribute to the loss and filter out the rest.
    rpn_class_logits = tf.gather_nd(rpn_class_logits, indices)
    anchor_class = tf.gather_nd(anchor_class, indices)

    # Cross entropy loss
    loss = K.sparse_categorical_crossentropy(target=anchor_class,
                                             output=rpn_class_logits,
                                             from_logits=True)
    loss = K.switch(tf.size(loss) > 0, K.mean(loss), tf.constant(0.0))

    return loss


def rpn_bbox_loss_graph(images_per_gpu, target_bbox, rpn_match, rpn_bbox):
    """
    Return the RPN bounding box loss graph.

    Args:
        images_per_gpu: the number of images by gpu
        target_bbox: [batch, max positive anchors, (dy, dx, dz, log(dh), log(dw), log(dd) )].
            Uses 0 padding to fill in unsed bbox deltas.
        rpn_match: [batch, anchors, 1]. Anchor match type. 1=positive,
                   -1=negative, 0=neutral anchor.
        rpn_bbox: [batch, anchors, (dy, dx, dz, log(dh), log(dw), log(dd) )]

    Returns:
        loss for RPN delta boxes
    """

    # Positive anchors contribute to the loss, but negative and
    # neutral anchors (match value of 0 or -1) don't.
    rpn_match = K.squeeze(rpn_match, -1)
    indices = tf.where(K.equal(rpn_match, 1))

    # Pick bbox deltas that contribute to the loss
    rpn_bbox = tf.gather_nd(rpn_bbox, indices)

    # Trim target bounding box deltas to the same length as rpn_bbox.
    batch_counts = K.sum(K.cast(K.equal(rpn_match, 1), tf.int32), axis=1)
    target_bbox = batch_pack_graph(target_bbox, batch_counts, images_per_gpu)

    loss = smooth_l1_loss(target_bbox, rpn_bbox)
    loss = K.switch(tf.size(loss) > 0, K.mean(loss), tf.constant(0.0))

    return loss


def mrcnn_class_loss_graph(target_class_ids, pred_class_logits,
                           active_class_ids):
    """
    Loss for the classifier head of Mask RCNN.

    target_class_ids: [batch, num_rois]. Integer class IDs. Uses zero
        padding to fill in the array.
    pred_class_logits: [batch, num_rois, num_classes]
    active_class_ids: [batch, num_classes]. Has a value of 1 for
        classes that are in the dataset of the image, and 0
        for classes that are not in the dataset.
    """

    # During model building, Keras calls this function with
    # target_class_ids of type float32. Unclear why. Cast it
    # to int to get around it.
    target_class_ids = tf.cast(target_class_ids, 'int64')

    # Find predictions of classes that are not in the dataset.
    pred_class_ids = tf.argmax(pred_class_logits, axis=2)
    pred_active = tf.gather(active_class_ids[0], pred_class_ids)

    # Loss
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=target_class_ids, 
        logits=pred_class_logits
    )

    # Erase losses of predictions of classes that are not in the active
    # classes of the image.
    loss = loss * pred_active

    # Computer loss mean. Use only predictions that contribute
    # to the loss to get a correct mean.
    loss = tf.reduce_sum(loss) / tf.reduce_sum(pred_active)

    return loss


def mrcnn_bbox_loss_graph(target_bbox, target_class_ids, pred_bbox):
    """
    Loss for Mask R-CNN bounding box refinement.

    target_bbox: [batch, num_rois, (dy, dx, log(dh), log(dw))]
    target_class_ids: [batch, num_rois]. Integer class IDs.
    pred_bbox: [batch, num_rois, num_classes, (dy, dx, log(dh), log(dw))]
    """

    # Reshape to merge batch and roi dimensions for simplicity.
    target_class_ids = K.reshape(target_class_ids, (-1,))
    target_bbox = K.reshape(target_bbox, (-1, 6))
    pred_bbox = K.reshape(pred_bbox, (-1, K.int_shape(pred_bbox)[2], 6))

    # Only positive ROIs contribute to the loss. And only
    # the right class_id of each ROI. Get their indices.
    positive_roi_ix = tf.where(target_class_ids > 0)[:, 0]
    positive_roi_class_ids = tf.cast(
        tf.gather(target_class_ids, positive_roi_ix), 
        tf.int64
    )
    indices = tf.stack([positive_roi_ix, positive_roi_class_ids], axis=1)

    # Gather the deltas (predicted and true) that contribute to loss
    target_bbox = tf.gather(target_bbox, positive_roi_ix)
    pred_bbox = tf.gather_nd(pred_bbox, indices)

    # Smooth-L1 Loss
    loss = K.switch(tf.size(target_bbox) > 0,
                    smooth_l1_loss(y_true=target_bbox, y_pred=pred_bbox),
                    tf.constant(0.0))
    loss = K.mean(loss)

    return loss


def mrcnn_mask_loss_graph(target_masks, target_class_ids, pred_masks):
    """
    Mask binary cross-entropy loss for the masks head.

    target_masks: [batch, num_rois, height, width, depth].
        A float32 tensor of values 0 or 1. Uses zero padding to fill array.
    target_class_ids: [batch, num_rois]. Integer class IDs. Zero padded.
    pred_masks: [batch, proposals, height, width, depth, num_classes] float32 tensor
                with values from 0 to 1.
    """

    # Reshape for simplicity. Merge first two dimensions into one.
    target_class_ids = K.reshape(target_class_ids, (-1,))
    mask_shape = tf.shape(target_masks)
    target_masks = K.reshape(target_masks, (-1, mask_shape[2], mask_shape[3], mask_shape[4]))
    pred_shape = tf.shape(pred_masks)
    pred_masks = K.reshape(pred_masks,
                           (-1, pred_shape[2], pred_shape[3], pred_shape[4], pred_shape[5]))
    
    # Permute predicted masks to [N, num_classes, height, width]
    pred_masks = tf.transpose(pred_masks, [0, 4, 1, 2, 3])

    # Only positive ROIs contribute to the loss. And only
    # the class specific mask of each ROI.
    positive_ix = tf.where(target_class_ids > 0)[:, 0]
    positive_class_ids = tf.cast(tf.gather(target_class_ids, positive_ix), tf.int64)
    indices = tf.stack([positive_ix, positive_class_ids], axis=1)

    # Gather the masks (predicted and true) that contribute to loss
    y_true = tf.gather(target_masks, positive_ix)
    y_pred = tf.gather_nd(pred_masks, indices)

    # Compute binary cross entropy. If no positive ROIs, then return 0.
    # shape: [batch, roi, num_classes]
    loss = K.switch(tf.size(y_true) > 0,
                    K.binary_crossentropy(target=y_true, output=y_pred),
                    tf.constant(0.0))
    loss = K.mean(loss)

    return loss


############################################################
#  Custom Callbacks
############################################################

class CheckPointCallback(keras.callbacks.Callback):
    def __init__(self, save_path, interval=1000):
        super(CheckPointCallback, self).__init__()
        self.save_path = save_path
        self.interval = interval
        self.counter = 0

    def on_batch_end(self):
        self.counter += 1
        if self.counter % self.interval == 0:
            self.model.save_weights(f"{self.save_path}checkpoint.h5")


class SaveWeightsCallback(keras.callbacks.Callback):
    def __init__(self, save_path):
        super(SaveWeightsCallback, self).__init__()
        self.save_path = save_path

    def on_epoch_end(self, epoch, logs=None):
        self.model.save_weights(f"{self.save_path}epoch_{str(epoch+1).zfill(3)}.h5")


class RPNEvaluationCallback(keras.callbacks.Callback):
    def __init__(self, model, config, train_dataset, valid_dataset, test_dataset, check_boxes=False):
        super(RPNEvaluationCallback, self).__init__()
        self.model = model
        self.config = config
        self.train_dataset = train_dataset
        self.valid_dataset = valid_dataset
        self.test_dataset = test_dataset
        self.check_boxes = check_boxes

    def on_epoch_end(self, epoch, logs=None):
        rpn_evaluation(self.model, self.config, ["TRAIN SUBSET", "VALID_SUBSET", "TEST SUBSET"], [self.train_dataset, self.valid_dataset, self.test_dataset], self.check_boxes)


class HeadEvaluationCallback(keras.callbacks.Callback):
    def __init__(self, model, config, train_dataset, valid_dataset, test_dataset):
        super(HeadEvaluationCallback, self).__init__()
        self.model = model
        self.config = config
        self.train_dataset = train_dataset
        self.valid_dataset = valid_dataset
        self.test_dataset = test_dataset

    def on_epoch_end(self, epoch, logs=None):
        head_evaluation(self.model, self.config, ["TRAIN SUBSET", "VALID SUBSET", "TEST SUBSET"], [self.train_dataset, self.valid_dataset, self.test_dataset])


class SegBranchEvaluationCallback(keras.callbacks.Callback):
    def __init__(self, model, config, train_dataset, valid_dataset, test_dataset):
        super(SegBranchEvaluationCallback, self).__init__()
        self.model = model
        self.config = config
        self.train_dataset = train_dataset
        self.valid_dataset = valid_dataset
        self.test_dataset = test_dataset

    def on_epoch_end(self, epoch, logs=None):
        segbranch_evaluation(self.model, self.config, ["TRAIN SUBSET", "VALID SUBSET", "TEST SUBSET"], [self.train_dataset, self.valid_dataset, self.test_dataset])


############################################################
#  Models
############################################################


class RPN():
    """
    Encapsulates the RPN Mask RCNN model functionality.

    The actual Keras model is in the keras_model property.
    """

    def __init__(self, config, show_summary):
        """
        config: RPN configuration
        weight_dir: training weight directory
        train_dataset: data.Dataset train object
        test_dataset: data.Dataset test object
        """

        assert config.MODE in ['training', 'targeting']

        self.config = config

        self.keras_model = self.build()
        
        self.epoch = self.config.FROM_EPOCH

        self.train_dataset, self.valid_dataset, self.test_dataset = self.prepare_datasets()

        if show_summary:
            self.print_summary()

    def prepare_datasets(self):

        # Create Datasets
        train_dataset = EmbryoDataset()
        train_dataset.load_dataset(data_dir=self.config.DATA_DIR, tag="train")
        train_dataset.prepare()

        valid_dataset = EmbryoDataset()
        valid_dataset.load_dataset(data_dir=self.config.DATA_DIR, tag="valid")
        valid_dataset.prepare()

        test_dataset = EmbryoDataset()
        test_dataset.load_dataset(data_dir=self.config.DATA_DIR, tag="test")
        test_dataset.prepare()

        return train_dataset, valid_dataset, test_dataset

    def print_summary(self):
        
        # Model summary
        self.keras_model.summary(line_length=140)

        # Number of example in Datasets
        print("\nTrain dataset contains:", len(self.train_dataset.image_info), " elements.")
        print("\nValid dataset contains:", len(self.valid_dataset.image_info), " elements.")
        print("Test dataset contains:", len(self.test_dataset.image_info), " elements.")

        # Configuration
        self.config.display()

    def build(self):
        """
        Build RPN Mask R-CNN architecture.
        """

        # Image size must be dividable by 2 multiple times
        h, w, d = self.config.IMAGE_SHAPE[:3]
        if h / 2 ** 6 != int(h / 2 ** 6) or w / 2 ** 6 != int(w / 2 ** 6) or d / 2 ** 6 != int(d / 2 ** 6):
            raise Exception("Image size must be dividable by 2 at least 6 times "
                            "to avoid fractions when downscaling and upscaling."
                            "For example, use 256, 320, 384, 448, 512, ... etc. ")

        # Inputs
        input_image = KL.Input(shape=self.config.IMAGE_SHAPE, name="input_image")
        input_image_meta = KL.Input(shape=[self.config.IMAGE_META_SIZE],name="input_image_meta")
        
        # RPN targets
        input_rpn_match = KL.Input(shape=[None, 1], name="input_rpn_match", dtype=tf.int32)
        input_rpn_bbox = KL.Input(shape=[None, 6], name="input_rpn_bbox", dtype=tf.float32)

        # CNN
        _, C2, C3, C4, C5 = resnet_graph(input_image, self.config.BACKBONE, stage5=True, train_bn=self.config.TRAIN_BN)
        
        # Top-down Layers
        P5 = KL.Conv3D(self.config.TOP_DOWN_PYRAMID_SIZE, (1, 1, 1), name='fpn_c5p5')(C5)

        P4 = KL.Add(name="fpn_p4add")([
            KL.UpSampling3D(size=(2, 2, 2), name="fpn_p5upsampled")(P5),
            KL.Conv3D(self.config.TOP_DOWN_PYRAMID_SIZE, (1, 1, 1), name='fpn_c4p4')(C4)
        ])

        P3 = KL.Add(name="fpn_p3add")([
            KL.UpSampling3D(size=(2, 2, 2), name="fpn_p4upsampled")(P4),
            KL.Conv3D(self.config.TOP_DOWN_PYRAMID_SIZE, (1, 1, 1), name='fpn_c3p3')(C3)
        ])
        P2 = KL.Add(name="fpn_p2add")([
            KL.UpSampling3D(size=(2, 2, 2), name="fpn_p3upsampled")(P3),
            KL.Conv3D(self.config.TOP_DOWN_PYRAMID_SIZE, (1, 1, 1), name='fpn_c2p2')(C2)
        ])
        
        # FPN last step
        P2 = KL.Conv3D(self.config.TOP_DOWN_PYRAMID_SIZE, (3, 3, 3), padding="SAME", name="fpn_p2")(P2)
        P3 = KL.Conv3D(self.config.TOP_DOWN_PYRAMID_SIZE, (3, 3, 3), padding="SAME", name="fpn_p3")(P3)
        P4 = KL.Conv3D(self.config.TOP_DOWN_PYRAMID_SIZE, (3, 3, 3), padding="SAME", name="fpn_p4")(P4)
        P5 = KL.Conv3D(self.config.TOP_DOWN_PYRAMID_SIZE, (3, 3, 3), padding="SAME", name="fpn_p5")(P5)
        P6 = KL.MaxPooling3D(pool_size=(1, 1, 1), strides=2, name="fpn_p6")(P5)

        # Features maps
        rpn_feature_maps = [P2, P3, P4, P5, P6]

        # Anchors
        anchors = self.get_anchors(self.config.IMAGE_SHAPE)
        # Duplicate across the batch dimension because Keras requires it
        anchors = np.broadcast_to(anchors, (self.config.BATCH_SIZE,) + anchors.shape)
        # A hack to get around Keras's bad support for constants
        anchors = KL.Lambda(lambda x: tf.Variable(anchors), name="anchors")(input_image)

        # RPN Model
        rpn = build_rpn_model(
            self.config.RPN_ANCHOR_STRIDE, 
            len(self.config.RPN_ANCHOR_RATIOS), 
            self.config.TOP_DOWN_PYRAMID_SIZE
        )

        # Loop through pyramid layers
        layer_outputs = []  # list of lists
        for p in rpn_feature_maps:
            layer_outputs.append(rpn([p]))

        # Concatenate layer outputs
        # Convert from list of lists of level outputs to list of lists
        # of outputs across levels.
        # e.g. [[a1, b1, c1], [a2, b2, c2]] => [[a1, a2], [b1, b2], [c1, c2]]
        output_names = ["rpn_class_logits", "rpn_class", "rpn_bbox"]
        outputs = list(zip(*layer_outputs))
        outputs = [KL.Concatenate(axis=1, name=n)(list(o)) for o, n in zip(outputs, output_names)]

        rpn_class_logits, rpn_class, rpn_bbox = outputs

        # Generate proposals
        rpn_rois = ProposalLayer(
            proposal_count=self.config.POST_NMS_ROIS_TRAINING,
            nms_threshold=self.config.RPN_NMS_THRESHOLD,
            pre_nms_limit=self.config.PRE_NMS_LIMIT,
            images_per_gpu=self.config.IMAGES_PER_GPU,
            rpn_bbox_std_dev=self.config.RPN_BBOX_STD_DEV,
            name="ROI"
        )([rpn_class, rpn_bbox, anchors])

        if self.config.MODE == "targeting":

            input_image_meta = KL.Input(shape=[self.config.IMAGE_META_SIZE], name="input_image_meta")
            input_gt_class_ids = KL.Input(shape=[None], name="input_gt_class_ids", dtype=tf.int32)

            input_gt_boxes = KL.Input(shape=[None, 6], name="input_gt_boxes", dtype=tf.float32)
            gt_boxes = KL.Lambda(lambda x: norm_boxes_graph(x, K.shape(input_image)[1:4]))(input_gt_boxes)

            if self.config.USE_MINI_MASK:
                input_gt_masks = KL.Input(shape=[*self.config.MINI_MASK_SHAPE, None], name="input_gt_masks", dtype=bool)
            else:
                input_gt_masks = KL.Input(shape=[*self.config.IMAGE_SHAPE[:-1], None], name="input_gt_masks", dtype=bool)

            rois, target_gt_boxes, target_class_ids, target_bbox, target_mask = DetectionTargetLayer(
                self.config.TRAIN_ROIS_PER_IMAGE,
                self.config.ROI_POSITIVE_RATIO,
                self.config.BBOX_STD_DEV,
                self.config.USE_MINI_MASK,
                self.config.MASK_SHAPE,
                self.config.IMAGES_PER_GPU,
                name="proposal_targets"
            )([rpn_rois, input_gt_class_ids, gt_boxes, input_gt_masks])

            rois_aligned = PyramidROIAlign(
                [self.config.POOL_SIZE, self.config.POOL_SIZE, self.config.POOL_SIZE],
                name="roi_align_classifier"
            )([rois, input_image_meta] + [P2, P3, P4, P5])
            
            mask_aligned = PyramidROIAlign(
                [self.config.MASK_POOL_SIZE, self.config.MASK_POOL_SIZE, self.config.MASK_POOL_SIZE],
                name="roi_align_mask"
            )([rois, input_image_meta] + [P2, P3, P4, P5])
            
            # Model
            inputs = [input_image, input_image_meta, input_gt_class_ids, input_gt_boxes, input_gt_masks]
            outputs = [rois, rois_aligned, mask_aligned, target_gt_boxes, target_class_ids, target_bbox, target_mask]

            model = KM.Model(inputs, outputs, name='rpn_targeting')

        else:

            rpn_class_loss = KL.Lambda(lambda x: rpn_class_loss_graph(*x), name="rpn_class_loss")(
                [input_rpn_match, rpn_class_logits])

            rpn_bbox_loss = KL.Lambda(lambda x: rpn_bbox_loss_graph(self.config.IMAGES_PER_GPU, *x), name="rpn_bbox_loss")(
                [input_rpn_bbox, input_rpn_match, rpn_bbox])
            
            # Model
            inputs = [input_image, input_rpn_match, input_rpn_bbox]
            outputs = [rpn_class_logits, rpn_class, rpn_bbox, rpn_rois, rpn_class_loss, rpn_bbox_loss]

            model = KM.Model(inputs, outputs, name='rpn_training')

        # Add multi-GPU support.
        if self.config.GPU_COUNT > 1:

            from core.parallel_model import ParallelModel
            model = ParallelModel(model, self.config.GPU_COUNT)

        return model
    
    def compile(self):
        """
        Gets the model ready for training. Adds losses, regularization, and
        metrics. Then calls the Keras compile() function.
        """

        self.keras_model.metrics_tensors = []

        if self.config.OPTIMIZER["name"] == "ADADELTA":

            optimizer = keras.optimizers.Adadelta(**self.config.OPTIMIZER["parameters"])

        elif self.config.OPTIMIZER["name"] == "SGD":

            optimizer = keras.optimizers.SGD(**self.config.OPTIMIZER["parameters"])

        # Add Losses
        # First, clear previously set losses to avoid duplication
        self.keras_model._losses = []
        self.keras_model._per_input_losses = {}
        loss_names = ["rpn_class_loss", "rpn_bbox_loss"]

        for name in loss_names:

            layer = self.keras_model.get_layer(name)
            loss = (
                    tf.reduce_mean(layer.output, keepdims=True)
                    * self.config.LOSS_WEIGHTS.get(name, 1.)
                )
            self.keras_model.add_loss(loss)
        
        # Add L2 Regularization
        # Skip gamma and beta weights of batch normalization layers.
        reg_losses = [
            keras.regularizers.l2(self.config.WEIGHT_DECAY)(w) / tf.cast(tf.size(w), tf.float32)
            for w in self.keras_model.trainable_weights
            if 'gamma' not in w.name and 'beta' not in w.name]
        self.keras_model.add_loss(tf.add_n(reg_losses))

        # Compile
        self.keras_model.compile(optimizer=optimizer, loss=[None] * len(self.keras_model.outputs))

    def train(self):

        assert self.config.MODE == "training", "Create model in training mode."

        # Create Data Generators
        train_generator = RPNGenerator(dataset=self.train_dataset, config=self.config)

        # Callback for saving weights
        save_weights = SaveWeightsCallback(self.config.WEIGHT_DIR)
        evaluation = RPNEvaluationCallback(self.keras_model, self.config, self.train_dataset, self.valid_dataset, self.test_dataset)
        
        # Model compilation
        self.compile()

        # Initialize weight dir
        os.makedirs(self.config.WEIGHT_DIR, exist_ok=True)

        # Load weights if self.config.RPN_WEIGHTS is not None
        if self.config.RPN_WEIGHTS:
            self.keras_model.load_weights(self.config.RPN_WEIGHTS, by_name=True)

        # Number of workers definition, the below values are optimized for our config
        # Work-around for Windows: Keras fails on Windows when using
        # multiprocessing workers. See discussion here:
        # https://github.com/matterport/Mask_RCNN/issues/13#issuecomment-353124009
        # if os.name is 'nt':
        #     workers = 0
        # else:
        #     workers = multiprocessing.cpu_count()

        # Training loop
        self.keras_model.fit_generator(
            train_generator,
            initial_epoch=self.config.FROM_EPOCH,
            epochs=self.config.FROM_EPOCH + self.config.EPOCHS,
            steps_per_epoch=len(self.train_dataset.image_info),
            callbacks=[save_weights, evaluation],
            validation_data=None,
            max_queue_size=30,
            workers=5,
            use_multiprocessing=True,
        )
    
    def get_anchors(self, image_shape):
        """Returns anchor pyramid for the given image size."""
        backbone_shapes = compute_backbone_shapes(self.config, image_shape)
        # Cache anchors and reuse if image shape is the same
        if not hasattr(self, "_anchor_cache"):
            self._anchor_cache = {}
        if not tuple(image_shape) in self._anchor_cache:
            # Generate Anchors
            a = utils.generate_pyramid_anchors(
                self.config.RPN_ANCHOR_SCALES,
                self.config.RPN_ANCHOR_RATIOS,
                backbone_shapes,
                self.config.BACKBONE_STRIDES,
                self.config.RPN_ANCHOR_STRIDE)
            # Normalize coordinates
            self._anchor_cache[tuple(image_shape)] = utils.norm_boxes(a, image_shape[:3])
        return self._anchor_cache[tuple(image_shape)]

    def head_target_generation(self):

        assert self.config.MODE == "targeting", "Create model in targeting mode."

        # Create Data Generators
        train_generator = RPNGenerator(dataset=self.train_dataset, config=self.config)
        valid_generator = RPNGenerator(dataset=self.valid_dataset, config=self.config)
        test_generator = RPNGenerator(dataset=self.test_dataset, config=self.config)

        # Create target folders
        base_path = f"{self.config.OUTPUT_DIR}"
        os.makedirs(base_path, exist_ok=True)
        os.makedirs(f"{base_path}datasets/", exist_ok=True)

        target_dirs = [
            f"{base_path}rois/",
            f"{base_path}rois_aligned/",
            f"{base_path}mask_aligned/",
            f"{base_path}target_class_ids/",
            f"{base_path}target_bbox/",
            f"{base_path}target_mask/",
        ]
        for target_dir in target_dirs:
            os.makedirs(target_dir, exist_ok=True)

        # Load RPN_WEIGHTS
        self.keras_model.load_weights(self.config.RPN_WEIGHTS, by_name=True)

        # Proper target generation
        for generator, set_type in zip([train_generator, valid_generator, test_generator], ["train", "valid", "test"]):
            
            print(f"TARGET GENERATION FOR {set_type} DATASET...")

            # Initiate example dataframe
            example_dataframe = pd.DataFrame({
                "rois": [], 
                "rois_aligned": [],
                "mask_aligned": [], 
                "target_class_ids": [],
                "target_bbox": [], 
                "target_mask": [],
            })

            # Loop over examples
            if set_type == "train":
                n = int(self.config.TARGET_RATIO * len(generator.image_ids))
            else:
                n = len(generator.image_ids)
            
            for ex_id in tqdm(range(n)):

                item_info = generator.dataset.image_info[ex_id]
                name = item_info["path"].split("/")[-1].split(".")[0]
                inputs, _ = generator.__getitem__(ex_id)

                # Note that outputs is [rois, rois_aligned, mask_aligned, _, target_class_ids, target_bbox, target_mask]
                outputs = self.keras_model.predict(inputs)
                # We don't want the fourth element
                del outputs[3]

                dataframe_info = []
                for target_dir, output in zip(target_dirs, outputs):

                    save_path = f"{target_dir}{name}.npy"

                    # Save example
                    np.save(save_path, output[0])

                    dataframe_info.append(save_path)

                example_dataframe.loc[len(example_dataframe.index)] = dataframe_info
            
            # Save dataset dataframe
            example_dataframe.to_csv(f"{base_path}datasets/{set_type}.csv", index=None)

    def evaluate(self):
        
        os.makedirs(f"{self.config.OUTPUT_DIR}RPN/boxes/", exist_ok=True)
        os.makedirs(f"{self.config.OUTPUT_DIR}RPN/seg_maps/", exist_ok=True)

        # Load RPN_WEIGHTS
        self.keras_model.load_weights(self.config.RPN_WEIGHTS, by_name=True)

        max_object_nb = self.config.MAX_GT_INSTANCES

        generator = RPNGenerator(self.test_dataset, self.config, shuffle=False)
        t = []
        for k in tqdm(range(self.config.EVALUATION_STEPS)):

            inputs, _ = generator.__getitem__(k)
            _, batch_rpn_class, _, batch_rpn_rois, _, _ = self.keras_model.predict(inputs)
            
            for m in range(batch_rpn_rois.shape[0]):

                _, boxes, _, _ = generator.load_image_gt(k * self.config.BATCH_SIZE + m)
                
                rpn_rois = denorm_boxes(batch_rpn_rois[m, :max_object_nb], self.config.IMAGE_SHAPE[:3])
                rpn_class = batch_rpn_class[m, :max_object_nb]
                rpn_class = rpn_class[:, 0]

                overlaps = compute_overlaps(boxes, rpn_rois)
                
                roi_association = -1 * np.ones(boxes.shape[0]).astype(np.uint8)
                box_association = -1 * np.ones(rpn_rois.shape[0]).astype(np.uint8)

                for j in range(rpn_rois.shape[0]):
                    roi_overlaps = overlaps[:, j]
                    argmax = np.argmax(roi_overlaps)
                    if roi_association[argmax] == -1:
                        if roi_overlaps[argmax] > 0.5:
                            roi_association[argmax] = j
                            box_association[j] = argmax
                
                roi_association = [indice for indice in roi_association if indice != -1]
                box_association = [indice for indice in box_association if indice != -1]

                box_association.sort()
                
                positive_rois = rpn_rois[roi_association]
                positive_boxes = boxes[box_association]

                detection_score = int(100 * positive_rois.shape[0] / boxes.shape[0])
                t.append(detection_score)

                pad = (np.sum(rpn_rois, axis=1) - 3).astype(bool)
                rpn_rois = rpn_rois[pad]
                nb_boxes = rpn_rois.shape[0]
                pd_boxes = positive_boxes.shape[0]
                print(detection_score, nb_boxes, pd_boxes)

                # image_path = self.test_dataset.image_info[k * self.config.BATCH_SIZE + m]["path"]
                # name = image_path.split("/")[-1][:-9]
                # gt_segs = imread(f"/data/icar/gdavid/Embryo3D/segs/{name}.tiff")

                # labels = np.unique(gt_segs)[1:]
                # labels = labels[box_association]

                # ious = (100 * compute_overlaps(positive_boxes, positive_rois).diagonal()).astype(np.uint8)
                # scores = (100 * rpn_class[roi_association]).astype(np.uint8)

                # score_segs = np.zeros(gt_segs.shape, dtype=np.uint8)
                # iou_segs = np.zeros(gt_segs.shape, dtype=np.uint8)
                # f = open(f"{self.config.OUTPUT_DIR}RPN/boxes/{detection_score}_{name}.dat", 'w')

                # for i, label in enumerate(labels):

                #     score_segs = np.where(gt_segs == label, scores[i], score_segs)
                #     iou_segs = np.where(gt_segs == label, ious[i], iou_segs)
                #     y1, x1, z1, y2, x2, z2 = np.split(positive_boxes[i], 6)
                #     y1p, x1p, z1p, y2p, x2p, z2p = np.split(positive_rois[i], 6)
                #     f.write('{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n'.format(ious[i], y1p, y1, x1p, x1, z1p, z1, y2p, y2, x2p, x2, z2p, z2))

                # seg_maps = np.concatenate([score_segs, iou_segs], axis=2)

                # imsave(f"{self.config.OUTPUT_DIR}RPN/seg_maps/{detection_score}_{name}.tiff", seg_maps)
                # f.close()
        print(np.mean(t))


class HEAD():
    """
    Encapsulates the Head Mask RCNN model functionality.

    The actual Keras model is in the keras_model property.
    """

    def __init__(self, config, show_summary):
        """
        config: RPN configuration
        weight_dir: training weight directory
        train_dataset: data.Dataset train object
        test_dataset: data.Dataset test object
        """
        
        self.config = config

        self.keras_model = self.build()
        
        self.epoch = self.config.FROM_EPOCH

        self.train_dataset, self.valid_dataset, self.test_dataset = self.prepare_datasets()

        if show_summary:
            self.print_summary()

    def prepare_datasets(self):

        # Create Datasets
        train_dataset = EmbryoHeadDataset()
        train_dataset.load_dataset(data_dir=self.config.DATA_DIR, tag="train")
        train_dataset.prepare()

        valid_dataset = EmbryoHeadDataset()
        valid_dataset.load_dataset(data_dir=self.config.DATA_DIR, tag="valid")
        valid_dataset.prepare()

        test_dataset = EmbryoHeadDataset()
        test_dataset.load_dataset(data_dir=self.config.DATA_DIR, tag="test")
        test_dataset.prepare()

        return train_dataset, valid_dataset, test_dataset

    def print_summary(self):
        
        # Model summary
        self.keras_model.summary(line_length=140)

        # Number of example in Datasets
        print("\nTrain dataset contains:", len(self.train_dataset.image_info), " elements.")
        print("\nValid dataset contains:", len(self.valid_dataset.image_info), " elements.")
        print("Test dataset contains:", len(self.test_dataset.image_info), " elements.")

        # Configuration
        self.config.display()

    def build(self):
        """
        Build Head Mask R-CNN architecture.
        """
        
        # Inputs
        input_rois_aligned = KL.Input(
            shape=[
                self.config.TRAIN_ROIS_PER_IMAGE, 
                self.config.POOL_SIZE, 
                self.config.POOL_SIZE, 
                self.config.POOL_SIZE,
                self.config.TOP_DOWN_PYRAMID_SIZE
            ], 
            name="input_rois_aligned"
        )

        input_image_meta = KL.Input(shape=[self.config.IMAGE_META_SIZE], name="input_image_meta")

        input_target_class_ids = KL.Input(shape=[self.config.TRAIN_ROIS_PER_IMAGE, ], name="input_target_class_ids")

        input_target_bbox = KL.Input(shape=[self.config.TRAIN_ROIS_PER_IMAGE, 6], name="input_target_bbox")

        input_mask_aligned = KL.Input(
            shape=[
                self.config.TRAIN_ROIS_PER_IMAGE, 
                self.config.MASK_POOL_SIZE, 
                self.config.MASK_POOL_SIZE,
                self.config.MASK_POOL_SIZE, 
                self.config.TOP_DOWN_PYRAMID_SIZE
            ],
            name="input_mask_aligned"
        )

        input_target_mask = KL.Input(
            shape=[self.config.TRAIN_ROIS_PER_IMAGE, *self.config.MASK_SHAPE, 1],
            name="input_target_mask"
        )

        active_class_ids = KL.Lambda(lambda x: parse_image_meta_graph(x)["active_class_ids"])(input_image_meta)

        # Network Heads: classifier and regressor
        mrcnn_class_logits, mrcnn_prob, mrcnn_bbox = fpn_classifier_graph(
            y=input_rois_aligned, 
            pool_size=self.config.POOL_SIZE,
            num_classes=self.config.NUM_CLASSES, 
            fc_layers_size=self.config.FPN_CLASSIF_FC_LAYERS_SIZE,
            train_bn=self.config.TRAIN_BN
        )
        
        # Network Heads: segmentation
        mrcnn_mask = build_fpn_mask_graph(
            y=input_mask_aligned, 
            num_classes=self.config.NUM_CLASSES,
            conv_channel=self.config.HEAD_CONV_CHANNEL, 
            train_bn=self.config.TRAIN_BN
        )

        # Losses
        bbox_loss = KL.Lambda(lambda x: mrcnn_bbox_loss_graph(*x), name="mrcnn_bbox_loss")(
            [input_target_bbox, input_target_class_ids, mrcnn_bbox])
        
        class_loss = KL.Lambda(lambda x: mrcnn_class_loss_graph(*x), name="mrcnn_class_loss")(
            [input_target_class_ids, mrcnn_class_logits, active_class_ids])
        
        mask_loss = KL.Lambda(lambda x: mrcnn_mask_loss_graph(*x), name="mrcnn_mask_loss")(
            [input_target_mask, input_target_class_ids, mrcnn_mask])

        # Model
        inputs = [input_rois_aligned, input_mask_aligned, input_image_meta, input_target_class_ids, input_target_bbox,
                  input_target_mask]
        outputs = [mrcnn_class_logits, mrcnn_prob, mrcnn_bbox, mrcnn_mask, class_loss, bbox_loss, mask_loss]

        model = KM.Model(inputs, outputs, name='head_training')

        # Add multi-GPU support.
        if self.config.GPU_COUNT > 1:

            from core.parallel_model import ParallelModel
            model = ParallelModel(model, self.config.GPU_COUNT)

        return model

    def compile(self,):
        """
        Gets the model ready for training. Adds losses, regularization, and
        metrics. Then calls the Keras compile() function.
        """

        self.keras_model.metrics_tensors = []

        if self.config.OPTIMIZER["name"] == "ADADELTA":

            optimizer = keras.optimizers.Adadelta(**self.config.OPTIMIZER["parameters"])

        elif self.config.OPTIMIZER["name"] == "SGD":

            optimizer = keras.optimizers.SGD(**self.config.OPTIMIZER["parameters"])

        # Add Losses
        # First, clear previously set losses to avoid duplication
        self.keras_model._losses = []
        self.keras_model._per_input_losses = {}
        loss_names = ["mrcnn_class_loss", "mrcnn_bbox_loss", "mrcnn_mask_loss"]

        for name in loss_names:

            layer = self.keras_model.get_layer(name)
            loss = (
                    tf.reduce_mean(layer.output, keepdims=True)
                    * self.config.LOSS_WEIGHTS.get(name, 1.)
                )
            self.keras_model.add_loss(loss)
        
        # Add L2 Regularization
        # Skip gamma and beta weights of batch normalization layers.
        reg_losses = [
            keras.regularizers.l2(self.config.WEIGHT_DECAY)(w) / tf.cast(tf.size(w), tf.float32)
            for w in self.keras_model.trainable_weights
            if 'gamma' not in w.name and 'beta' not in w.name]
        self.keras_model.add_loss(tf.add_n(reg_losses))

        # Compile
        self.keras_model.compile(optimizer=optimizer, loss=[None] * len(self.keras_model.outputs))

        # Add metrics for losses
        for name in loss_names:
            if name in self.keras_model.metrics_names:
                continue
            layer = self.keras_model.get_layer(name)
            self.keras_model.metrics_names.append(name)
            loss = (layer.output * self.config.LOSS_WEIGHTS.get(name, 1.))
            self.keras_model.metrics_tensors.append(loss)

    def train(self):
        
        assert self.config.MODE == "training", "Create model in training mode."

        # Create Data Generators
        train_generator = HeadGenerator(dataset=self.train_dataset, config=self.config)

        # Callback for saving weights
        save_weights = SaveWeightsCallback(self.config.WEIGHT_DIR)
        evaluation = HeadEvaluationCallback(self.keras_model, self.config, self.train_dataset, self.valid_dataset, self.test_dataset)
        
        # Model compilation
        self.compile()

        # Initialize weight dir
        os.makedirs(self.config.WEIGHT_DIR, exist_ok=True)

        # Load weights if self.config.HEAD_WEIGHTS is not None
        if self.config.HEAD_WEIGHTS:
            self.keras_model.load_weights(self.config.HEAD_WEIGHTS, by_name=True)

        # Work-around for Windows: Keras fails on Windows when using
        # multiprocessing workers. See discussion here:
        # https://github.com/matterport/Mask_RCNN/issues/13#issuecomment-353124009
        if os.name == 'nt':
            workers = 0
        else:
            workers = multiprocessing.cpu_count()

        # Training loop
        self.keras_model.fit_generator(
            train_generator,
            initial_epoch=self.config.FROM_EPOCH,
            epochs=self.config.FROM_EPOCH + self.config.EPOCHS,
            steps_per_epoch=len(self.train_dataset.image_info),
            callbacks=[save_weights, evaluation],
            validation_data=None,
            max_queue_size=10,
            workers=workers,
            use_multiprocessing=True,
        )


class SegBranch():
    """
    Encapsulates the Head Mask RCNN model functionality.

    The actual Keras model is in the keras_model property.
    """

    def __init__(self, config, show_summary):
        """
        config: RPN configuration
        weight_dir: training weight directory
        train_dataset: data.Dataset train object
        test_dataset: data.Dataset test object
        """
        
        self.config = config

        self.keras_model = self.build()
        
        self.epoch = self.config.FROM_EPOCH

        self.train_dataset, self.valid_dataset, self.test_dataset = self.prepare_datasets()

        if show_summary:
            self.print_summary()

    def prepare_datasets(self):

        # Create Datasets
        train_dataset = EmbryoHeadDataset()
        train_dataset.load_dataset(data_dir=self.config.DATA_DIR, tag="train")
        train_dataset.prepare()

        valid_dataset = EmbryoHeadDataset()
        valid_dataset.load_dataset(data_dir=self.config.DATA_DIR, tag="valid")
        valid_dataset.prepare()

        test_dataset = EmbryoHeadDataset()
        test_dataset.load_dataset(data_dir=self.config.DATA_DIR, tag="test")
        test_dataset.prepare()

        return train_dataset, valid_dataset, test_dataset

    def print_summary(self):
        
        # Model summary
        self.keras_model.summary(line_length=140)

        # Number of example in Datasets
        print("\nTrain dataset contains:", len(self.train_dataset.image_info), " elements.")
        print("\nValid dataset contains:", len(self.valid_dataset.image_info), " elements.")
        print("Test dataset contains:", len(self.test_dataset.image_info), " elements.")

        # Configuration
        self.config.display()

    def build(self):
        """
        Build Head Mask R-CNN architecture.
        """
        
        # Inputs
        input_target_class_ids = KL.Input(shape=[self.config.TRAIN_ROIS_PER_IMAGE, ], name="input_target_class_ids")

        input_mask_aligned = KL.Input(
            shape=[
                self.config.TRAIN_ROIS_PER_IMAGE, 
                self.config.MASK_POOL_SIZE, 
                self.config.MASK_POOL_SIZE,
                self.config.MASK_POOL_SIZE, 
                self.config.TOP_DOWN_PYRAMID_SIZE
            ],
            name="input_mask_aligned"
        )

        input_target_mask = KL.Input(
            shape=[self.config.TRAIN_ROIS_PER_IMAGE, *self.config.MASK_SHAPE, 1],
            name="input_target_mask"
        )
        
        # Network Heads: segmentation
        mrcnn_mask = build_fpn_mask_graph(
            y=input_mask_aligned, 
            num_classes=self.config.NUM_CLASSES,
            conv_channel=self.config.HEAD_CONV_CHANNEL, 
            train_bn=self.config.TRAIN_BN
        )
        
        mask_loss = KL.Lambda(lambda x: mrcnn_mask_loss_graph(*x), name="mrcnn_mask_loss")(
            [input_target_mask, input_target_class_ids, mrcnn_mask])

        # Model
        inputs = [input_mask_aligned, input_target_class_ids, input_target_mask]
        outputs = [mrcnn_mask, mask_loss]

        model = KM.Model(inputs, outputs, name='segbranch_training')

        # Add multi-GPU support.
        if self.config.GPU_COUNT > 1:

            from core.parallel_model import ParallelModel
            model = ParallelModel(model, self.config.GPU_COUNT)

        return model

    def compile(self,):
        """
        Gets the model ready for training. Adds losses, regularization, and
        metrics. Then calls the Keras compile() function.
        """

        self.keras_model.metrics_tensors = []

        if self.config.OPTIMIZER["name"] == "ADADELTA":

            optimizer = keras.optimizers.Adadelta(**self.config.OPTIMIZER["parameters"])

        elif self.config.OPTIMIZER["name"] == "SGD":

            optimizer = keras.optimizers.SGD(**self.config.OPTIMIZER["parameters"])

        # Add Losses
        # First, clear previously set losses to avoid duplication
        self.keras_model._losses = []
        self.keras_model._per_input_losses = {}
        loss_names = ["mrcnn_mask_loss"]

        for name in loss_names:

            layer = self.keras_model.get_layer(name)
            loss = (
                    tf.reduce_mean(layer.output, keepdims=True)
                    * self.config.LOSS_WEIGHTS.get(name, 1.)
                )
            self.keras_model.add_loss(loss)
        
        # Add L2 Regularization
        # Skip gamma and beta weights of batch normalization layers.
        reg_losses = [
            keras.regularizers.l2(self.config.WEIGHT_DECAY)(w) / tf.cast(tf.size(w), tf.float32)
            for w in self.keras_model.trainable_weights
            if 'gamma' not in w.name and 'beta' not in w.name]
        self.keras_model.add_loss(tf.add_n(reg_losses))

        # Compile
        self.keras_model.compile(optimizer=optimizer, loss=[None] * len(self.keras_model.outputs))

        # Add metrics for losses
        for name in loss_names:
            if name in self.keras_model.metrics_names:
                continue
            layer = self.keras_model.get_layer(name)
            self.keras_model.metrics_names.append(name)
            loss = (layer.output * self.config.LOSS_WEIGHTS.get(name, 1.))
            self.keras_model.metrics_tensors.append(loss)

    def train(self):
        
        assert self.config.MODE == "training", "Create model in training mode."

        # Create Data Generators
        train_generator = SegBranchGenerator(dataset=self.train_dataset, config=self.config)

        # Callback for saving weights
        save_weights = SaveWeightsCallback(self.config.WEIGHT_DIR)
        evaluation = SegBranchEvaluationCallback(self.keras_model, self.config, self.train_dataset, self.valid_dataset, self.test_dataset)
        
        # Model compilation
        self.compile()

        # Initialize weight dir
        os.makedirs(self.config.WEIGHT_DIR, exist_ok=True)

        # Load weights if self.config.HEAD_WEIGHTS is not None
        if self.config.HEAD_WEIGHTS:
            self.keras_model.load_weights(self.config.HEAD_WEIGHTS, by_name=True)

        # Work-around for Windows: Keras fails on Windows when using
        # multiprocessing workers. See discussion here:
        # https://github.com/matterport/Mask_RCNN/issues/13#issuecomment-353124009
        if os.name == 'nt':
            workers = 0
        else:
            workers = multiprocessing.cpu_count()

        # Training loop
        self.keras_model.fit_generator(
            train_generator,
            initial_epoch=self.config.FROM_EPOCH,
            epochs=self.config.FROM_EPOCH + self.config.EPOCHS,
            steps_per_epoch=len(self.train_dataset.image_info),
            callbacks=[save_weights, evaluation],
            validation_data=None,
            max_queue_size=10,
            workers=workers,
            use_multiprocessing=True,
        )


class MaskRCNN():
    """Encapsulates the whole Mask RCNN model functionality.

    The actual Keras model is in the keras_model property.
    """

    def __init__(self, config, show_summary):
        """
        mode: Either "training" or "inference"
        config: A Sub-class of the Config class
        model_dir: Directory to save training logs and trained weights
        """
        assert config.MODE in ['training', 'inference']
        
        self.config = config

        self.keras_model = self.build()
        
        self.epoch = self.config.FROM_EPOCH

        self.train_dataset, self.valid_dataset, self.test_dataset = self.prepare_datasets()

        if show_summary:
            self.print_summary()

    def prepare_datasets(self):

        # Create Datasets
        train_dataset = EmbryoDataset()
        train_dataset.load_dataset(data_dir=self.config.DATA_DIR, tag="train")
        train_dataset.prepare()

        valid_dataset = EmbryoDataset()
        valid_dataset.load_dataset(data_dir=self.config.DATA_DIR, tag="valid")
        valid_dataset.prepare()

        test_dataset = EmbryoDataset()
        test_dataset.load_dataset(data_dir=self.config.DATA_DIR, tag="test")
        test_dataset.prepare()

        return train_dataset, valid_dataset, test_dataset

    def print_summary(self):
        
        # Model summary
        self.keras_model.summary(line_length=140)

        # Number of example in Datasets
        print("\nTrain dataset contains:", len(self.train_dataset.image_info), " elements.")
        print("\nValid dataset contains:", len(self.valid_dataset.image_info), " elements.")
        print("Test dataset contains:", len(self.test_dataset.image_info), " elements.")

        # Configuration
        self.config.display()

    def build(self):
        """
        Build Mask R-CNN architecture.
        """

        assert self.config.MODE in ['training', 'inference']
        
        # Image size must be dividable by 2 multiple times
        h, w, d = self.config.IMAGE_SHAPE[:3]
        if h / 2 ** 6 != int(h / 2 ** 6) or w / 2 ** 6 != int(w / 2 ** 6) or d / 2 ** 6 != int(d / 2 ** 6):
            raise Exception("Image size must be dividable by 2 at least 6 times "
                            "to avoid fractions when downscaling and upscaling."
                            "For example, use 256, 320, 384, 448, 512, ... etc. ")

        # Inputs
        input_image = KL.Input(shape=[*self.config.IMAGE_SHAPE], name="input_image")
        input_image_meta = KL.Input(shape=[self.config.IMAGE_META_SIZE], name="input_image_meta")

        if self.config.MODE == "training":
            
            # RPN targets
            input_rpn_match = KL.Input(shape=[None, 1], name="input_rpn_match", dtype=tf.int32)
            input_rpn_bbox = KL.Input(shape=[None, 6], name="input_rpn_bbox", dtype=tf.float32)

            input_gt_class_ids = KL.Input(shape=[None], name="input_gt_class_ids", dtype=tf.int32)

            input_gt_boxes = KL.Input(shape=[None, 6], name="input_gt_boxes", dtype=tf.float32)
            gt_boxes = KL.Lambda(lambda x: norm_boxes_graph(x, K.shape(input_image)[1:4]))(input_gt_boxes)

            if self.config.USE_MINI_MASK:
                input_gt_masks = KL.Input(shape=[*self.config.MINI_MASK_SHAPE, None], name="input_gt_masks", dtype=bool)
            else:
                input_gt_masks = KL.Input(shape=[*self.config.IMAGE_SHAPE[:-1], None], name="input_gt_masks", dtype=bool)

        # CNN
        _, C2, C3, C4, C5 = resnet_graph(input_image, self.config.BACKBONE, stage5=True, train_bn=self.config.TRAIN_BN)

        # Top-down Layers
        P5 = KL.Conv3D(self.config.TOP_DOWN_PYRAMID_SIZE, (1, 1, 1), name='fpn_c5p5')(C5)

        P4 = KL.Add(name="fpn_p4add")([
            KL.UpSampling3D(size=(2, 2, 2), name="fpn_p5upsampled")(P5),
            KL.Conv3D(self.config.TOP_DOWN_PYRAMID_SIZE, (1, 1, 1), name='fpn_c4p4')(C4)
        ])

        P3 = KL.Add(name="fpn_p3add")([
            KL.UpSampling3D(size=(2, 2, 2), name="fpn_p4upsampled")(P4),
            KL.Conv3D(self.config.TOP_DOWN_PYRAMID_SIZE, (1, 1, 1), name='fpn_c3p3')(C3)
        ])
        P2 = KL.Add(name="fpn_p2add")([
            KL.UpSampling3D(size=(2, 2, 2), name="fpn_p3upsampled")(P3),
            KL.Conv3D(self.config.TOP_DOWN_PYRAMID_SIZE, (1, 1, 1), name='fpn_c2p2')(C2)
        ])

        # FPN last step
        P2 = KL.Conv3D(self.config.TOP_DOWN_PYRAMID_SIZE, (3, 3, 3), padding="SAME", name="fpn_p2")(P2)
        P3 = KL.Conv3D(self.config.TOP_DOWN_PYRAMID_SIZE, (3, 3, 3), padding="SAME", name="fpn_p3")(P3)
        P4 = KL.Conv3D(self.config.TOP_DOWN_PYRAMID_SIZE, (3, 3, 3), padding="SAME", name="fpn_p4")(P4)
        P5 = KL.Conv3D(self.config.TOP_DOWN_PYRAMID_SIZE, (3, 3, 3), padding="SAME", name="fpn_p5")(P5)
        P6 = KL.MaxPooling3D(pool_size=(1, 1, 1), strides=2, name="fpn_p6")(P5)

        # Features maps
        # Note that P6 is used in RPN, but not in the classifier heads.
        rpn_feature_maps = [P2, P3, P4, P5, P6]
        mrcnn_feature_maps = [P2, P3, P4, P5]

        # Anchors
        if self.config.MODE == "training":

            anchors = self.get_anchors(self.config.IMAGE_SHAPE)
            # Duplicate across the batch dimension because Keras requires it
            anchors = np.broadcast_to(anchors, (self.config.BATCH_SIZE,) + anchors.shape)
            # A hack to get around Keras's bad support for constants
            anchors = KL.Lambda(lambda x: tf.Variable(anchors), name="anchors")(input_image)

        elif self.config.MODE == "inference":

            input_anchors = KL.Input(shape=[None, 6], name="input_anchors")
            anchors = input_anchors

        # RPN Model
        rpn = build_rpn_model(
            self.config.RPN_ANCHOR_STRIDE, 
            len(self.config.RPN_ANCHOR_RATIOS), 
            self.config.TOP_DOWN_PYRAMID_SIZE
        )

        # Loop through pyramid layers
        layer_outputs = []  # list of lists
        for p in rpn_feature_maps:
            layer_outputs.append(rpn([p]))
        
        # Concatenate layer outputs
        # Convert from list of lists of level outputs to list of lists
        # of outputs across levels.
        # e.g. [[a1, b1, c1], [a2, b2, c2]] => [[a1, a2], [b1, b2], [c1, c2]]
        output_names = ["rpn_class_logits", "rpn_class", "rpn_bbox"]
        outputs = list(zip(*layer_outputs))
        outputs = [KL.Concatenate(axis=1, name=n)(list(o)) for o, n in zip(outputs, output_names)]

        rpn_class_logits, rpn_class, rpn_bbox = outputs

        # Generate proposals
        proposal_count = self.config.POST_NMS_ROIS_TRAINING if self.config.MODE == "training" else self.config.POST_NMS_ROIS_INFERENCE
        rpn_rois = ProposalLayer(
            proposal_count=proposal_count,
            nms_threshold=self.config.RPN_NMS_THRESHOLD,
            pre_nms_limit=self.config.PRE_NMS_LIMIT,
            images_per_gpu=self.config.IMAGES_PER_GPU,
            rpn_bbox_std_dev=self.config.RPN_BBOX_STD_DEV,
            name="ROI"
        )([rpn_class, rpn_bbox, anchors])

        if self.config.MODE == "training":

            active_class_ids = KL.Lambda(lambda x: parse_image_meta_graph(x)["active_class_ids"])(input_image_meta)

            # Generate detection targets
            rois, _, target_class_ids, target_bbox, target_mask = DetectionTargetLayer(
                self.config.TRAIN_ROIS_PER_IMAGE,
                self.config.ROI_POSITIVE_RATIO,
                self.config.BBOX_STD_DEV,
                self.config.USE_MINI_MAKS,
                self.config.MASK_SHAPE,
                self.config.IMAGES_PER_GPU,
                name="proposal_targets"
            )([rpn_rois, input_gt_class_ids, gt_boxes, input_gt_masks])

            # Network Heads: classifier and regressor
            mrcnn_class_logits, mrcnn_class, mrcnn_bbox = fpn_classifier_graph_with_RoiAlign(
                rois=rois, 
                feature_maps=mrcnn_feature_maps, 
                image_meta=input_image_meta, 
                pool_size=self.config.POOL_SIZE,
                num_classes=self.config.NUM_CLASSES, 
                fc_layers_size=self.config.FPN_CLASSIF_FC_LAYERS_SIZE,
                train_bn=self.config.TRAIN_BN                
            )

            # Network Heads: segmentation
            mrcnn_mask = build_fpn_mask_graph_with_RoiAlign(
                rois=rois, 
                feature_maps=mrcnn_feature_maps, 
                image_meta=input_image_meta,
                pool_size=self.config.MASK_POOL_SIZE, 
                num_classes=self.config.NUM_CLASSES,
                conv_channel=self.config.HEAD_CONV_CHANNEL,
                train_bn=self.config.TRAIN_BN
            )

            output_rois = KL.Lambda(lambda x: x * 1, name="output_rois")(rois)

            # Losses
            rpn_class_loss = KL.Lambda(lambda x: rpn_class_loss_graph(*x), name="rpn_class_loss")(
                [input_rpn_match, rpn_class_logits])
            
            rpn_bbox_loss = KL.Lambda(lambda x: rpn_bbox_loss_graph(self.config.IMAGES_PER_GPU, *x), name="rpn_bbox_loss")(
                [input_rpn_bbox, input_rpn_match, rpn_bbox])
            
            class_loss = KL.Lambda(lambda x: mrcnn_class_loss_graph(*x), name="mrcnn_class_loss")(
                [target_class_ids, mrcnn_class_logits, active_class_ids])
            
            bbox_loss = KL.Lambda(lambda x: mrcnn_bbox_loss_graph(*x), name="mrcnn_bbox_loss")(
                [target_bbox, target_class_ids, mrcnn_bbox])
            
            mask_loss = KL.Lambda(lambda x: mrcnn_mask_loss_graph(*x), name="mrcnn_mask_loss")(
                [target_mask, target_class_ids, mrcnn_mask])

            # Model
            inputs = [input_image, input_image_meta, input_gt_class_ids, input_gt_boxes, input_gt_masks,
                      input_rpn_match, input_rpn_bbox]
            outputs = [rpn_class_logits, rpn_class, rpn_bbox, mrcnn_class_logits, mrcnn_class, mrcnn_bbox, mrcnn_mask,
                       rpn_rois, output_rois, rpn_class_loss, rpn_bbox_loss, class_loss, bbox_loss, mask_loss]
            
            model = KM.Model(inputs, outputs, name='mask_rcnn_training')

        elif self.config.MODE == "inference":

            # Network Heads: classifier and regressor
            mrcnn_class_logits, mrcnn_class, mrcnn_bbox = fpn_classifier_graph_with_RoiAlign(
                rois=rpn_rois, 
                feature_maps=mrcnn_feature_maps, 
                image_meta=input_image_meta, 
                pool_size=self.config.POOL_SIZE,
                num_classes=self.config.NUM_CLASSES, 
                fc_layers_size=self.config.FPN_CLASSIF_FC_LAYERS_SIZE,
                train_bn=self.config.TRAIN_BN
            )

            # Detections
            # output is [batch, num_detections, (y1, x1, z1, y2, x2, z2, class_id, score)] in
            # normalized coordinates
            detections = DetectionLayer(
                self.config.BBOX_STD_DEV,
                self.config.DETECTION_MIN_CONFIDENCE,
                self.config.DETECTION_MAX_INSTANCES,
                self.config.DETECTION_NMS_THRESHOLD,
                self.config.IMAGES_PER_GPU,
                self.config.BATCH_SIZE,
                name="mrcnn_detection"
            )([rpn_rois, mrcnn_class, mrcnn_bbox, input_image_meta])

            detection_boxes = KL.Lambda(lambda x: x[..., :6])(detections)

            # Create masks for detections
            mrcnn_mask = build_fpn_mask_graph_with_RoiAlign(
                rois=detection_boxes, 
                feature_maps=mrcnn_feature_maps, 
                image_meta=input_image_meta,
                pool_size=self.config.MASK_POOL_SIZE, 
                num_classes=self.config.NUM_CLASSES,
                conv_channel=self.config.HEAD_CONV_CHANNEL,
                train_bn=self.config.TRAIN_BN
            )

            # Model
            inputs = [input_image, input_image_meta, input_anchors]
            outputs = [detections, mrcnn_class, mrcnn_bbox, mrcnn_mask, rpn_rois, rpn_class, rpn_bbox]

            model = KM.Model(inputs, outputs, name='mask_rcnn_inference')

        # Add multi-GPU support.
        if self.config.GPU_COUNT > 1:

            from core.parallel_model import ParallelModel
            model = ParallelModel(model, self.config.GPU_COUNT)

        return model

    def compile(self):
        """
        Gets the model ready for training. Adds losses, regularization, and
        metrics. Then calls the Keras compile() function.
        """

        self.keras_model.metrics_tensors = []

        if self.config.OPTIMIZER["name"] == "ADADELTA":

            optimizer = keras.optimizers.Adadelta(**self.config.OPTIMIZER["parameters"])

        elif self.config.OPTIMIZER["name"] == "SGD":

            optimizer = keras.optimizers.SGD(**self.config.OPTIMIZER["parameters"])

        # Add Losses
        # First, clear previously set losses to avoid duplication
        self.keras_model._losses = []
        self.keras_model._per_input_losses = {}

        if self.config.LEARNING_LAYERS == "rpn":

            loss_names = ["rpn_class_loss", "rpn_bbox_loss"]

        elif self.config.LEARNING_LAYERS == "heads":

            loss_names = [
                "mrcnn_class_loss", "mrcnn_bbox_loss", "mrcnn_mask_loss"]
            
        elif self.config.LEARNING_LAYERS == "all":

            loss_names = [
                "rpn_class_loss", "rpn_bbox_loss",
                "mrcnn_class_loss", "mrcnn_bbox_loss", "mrcnn_mask_loss"]
            
        for name in loss_names:
            layer = self.keras_model.get_layer(name)
            loss = (
                    tf.reduce_mean(layer.output, keepdims=True)
                    * self.config.LOSS_WEIGHTS.get(name, 1.)
                )
            self.keras_model.add_loss(loss)

        # Add L2 Regularization
        # Skip gamma and beta weights of batch normalization layers.
        reg_losses = [
            keras.regularizers.l2(self.config.WEIGHT_DECAY)(w) / tf.cast(tf.size(w), tf.float32)
            for w in self.keras_model.trainable_weights
            if 'gamma' not in w.name and 'beta' not in w.name]
        self.keras_model.add_loss(tf.add_n(reg_losses))

        # Compile
        self.keras_model.compile(optimizer=optimizer, loss=[None] * len(self.keras_model.outputs))

    def set_trainable(self, layer_regex, keras_model=None, indent=0, verbose=1):
        """
        Sets model layers as trainable if their names match
        the given regular expression.
        """

        keras_model = keras_model or self.keras_model

        # In multi-GPU training, we wrap the model. Get layers
        # of the inner model because they have the weights.
        layers = self.keras_model.inner_model.layers if hasattr(self.keras_model, "inner_model") \
            else self.keras_model.layers

        for layer in layers:
            # Is the layer a model?
            if layer.__class__.__name__ == 'Model':
                print("In model: ", layer.name)
                self.set_trainable(
                    layer_regex, keras_model=layer, indent=indent + 4)
                continue

            if not layer.weights:
                continue
            # Is it trainable?
            trainable = bool(re.fullmatch(layer_regex, layer.name))
            # Update layer. If layer is a container, update inner layer.
            if layer.__class__.__name__ == 'TimeDistributed':
                layer.layer.trainable = trainable
            else:
                layer.trainable = trainable
                
    def train(self):

        assert self.config.MODE == "training", "Create model in training mode."

        # Create Data Generators
        train_generator = MrcnnGenerator(dataset=self.train_dataset, config=self.config)

        # Callback for saving weights
        save_weights = SaveWeightsCallback(self.config.WEIGHT_DIR)

        # Set only self.config.LEARNING_LAYERS trainable
        layers = self.config.LEARNING_LAYERS
        layer_regex = {
            # rpn
            "rpn": r"(conv1)|(bn_conv1)|(res2.*)|(bn2.*)|(res3.*)|(bn3.*)|(res4.*)|(bn4.*)|(res5.*)|(bn5.*)|(rpn\_.*)|(fpn\_.*)",
            # all layers but the backbone
            "heads": r"(mrcnn\_.*)",
            # All layers
            "all": ".*",
        }

        if layers in layer_regex.keys():
            layers = layer_regex[layers]

        self.set_trainable(layers)

        # Model compilation
        self.compile()

        # Initialize weight dir
        os.makedirs(self.config.WEIGHT_DIR, exist_ok=True)

        # These conditions allows to customize the weights imports
        if self.config.MASK_WEIGHTS:
            self.keras_model.load_weights(self.config.MASK_WEIGHTS, by_name=True)
        if self.config.RPN_WEIGHTS:
            self.keras_model.load_weights(self.config.RPN_WEIGHTS, by_name=True)
        if self.config.HEAD_WEIGHTS:
            self.keras_model.load_weights(self.config.HEAD_WEIGHTS, by_name=True)

        # Work-around for Windows: Keras fails on Windows when using
        # multiprocessing workers. See discussion here:
        # https://github.com/matterport/Mask_RCNN/issues/13#issuecomment-353124009
        if os.name == 'nt':
            workers = 0
        else:
            workers = multiprocessing.cpu_count()

        self.keras_model.fit_generator(
            train_generator,
            initial_epoch=self.config.FROM_EPOCH,
            epochs=self.config.FROM_EPOCH + self.config.EPOCHS,
            steps_per_epoch=len(self.train_dataset.image_info),
            callbacks=[save_weights],
            validation_data=None,
            max_queue_size=30,
            workers=workers,
            use_multiprocessing=True,
        )

    def evaluate(self):

        assert self.config.MODE == "inference", "Create model in inference mode."

        # These conditions allows to customize the weights imports
        if self.config.MASK_WEIGHTS:
            self.keras_model.load_weights(self.config.MASK_WEIGHTS, by_name=True)
        if self.config.RPN_WEIGHTS:
            self.keras_model.load_weights(self.config.RPN_WEIGHTS, by_name=True)
        if self.config.HEAD_WEIGHTS:
            self.keras_model.load_weights(self.config.HEAD_WEIGHTS, by_name=True)

        # Create Data Generators
        data_generator = MrcnnGenerator(dataset=self.test_dataset, config=self.config)

        result_dataframe = pd.DataFrame({
        "name": [],
        "instance_nb": [],
        "map-50": [],
        "precision-50": [],
        "recall-50": [],
        "iou-50": [],
        })

        result_dir = self.config.OUTPUT_DIR
        os.makedirs(result_dir, exist_ok=True)
        os.makedirs(f"{result_dir}pred/", exist_ok=True)
        os.makedirs(f"{result_dir}gt/", exist_ok=True)

        for i in tqdm(range(len(self.test_dataset.image_info))):
            # Load inputs
            name, inputs = data_generator.get_input_prediction(i)

            # Load ground truth
            _, _, gt_boxes, _, gt_masks = data_generator.load_image_gt(i)

            # Raw prediction
            detections, _, _, mrcnn_mask, _, _, _ = self.keras_model.predict(inputs)

            # Unmold prediction
            pd_boxes, _, _, pd_segs = self.unmold_detections(detections[0], mrcnn_mask[0])
            # gt_segs = self.unmold_groundtruth(gt_boxes, gt_masks)
            gt_segs = imread(f"/data/icar/gdavid/Embryo3D/segs/{name[:-9]}.tiff")

            print(name, len(np.unique(gt_segs)), len(np.unique(pd_segs)))

            # Save predicted instance segmentation
            imsave(f"{result_dir}pred/{name}", pd_segs.astype(np.uint16), check_contrast=False)
            imsave(f"{result_dir}gt/{name}", gt_segs.astype(np.uint16), check_contrast=False)

            # Evaluate
            # map50, precision50, recall50, ious = compute_ap(gt_boxes, gt_segs, pd_boxes, pd_segs, iou_threshold=0.5)
            # print(map50, precision50, recall50, ious)

            # Write results in dataframe
            # result_dataframe.loc[len(result_dataframe.index)] = [name, gt_masks.shape[-1], map50, precision50, recall50, np.mean(ious)]

        # Save dataframe
        # result_dataframe.to_csv(f"{result_dir}report.csv", index=None)

        # Print mean results
        # print(result_dataframe.mean())
    
    def head_evaluate(self):

        assert self.config.MODE == "inference", "Create model in inference mode."

        # These conditions allows to customize the weights imports
        if self.config.MASK_WEIGHTS:
            self.keras_model.load_weights(self.config.MASK_WEIGHTS, by_name=True)
        if self.config.RPN_WEIGHTS:
            self.keras_model.load_weights(self.config.RPN_WEIGHTS, by_name=True)
        if self.config.HEAD_WEIGHTS:
            self.keras_model.load_weights(self.config.HEAD_WEIGHTS, by_name=True)

        # Create Data Generators
        data_generator = MrcnnGenerator(dataset=self.test_dataset, config=self.config)

        # for i in tqdm(range(len(self.test_dataset.image_info))):
        for i in tqdm(range(1)):
            # Load inputs
            name, inputs = data_generator.get_input_prediction(i)

            # Load ground truth
            _, _, gt_boxes, _, _ = data_generator.load_image_gt(i)

            # Raw prediction
            _, mrcnn_class, _, _, rpn_rois, _, _ = self.keras_model.predict(inputs)

            mrcnn_class = np.argmax(mrcnn_class[0], axis=-1).astype(np.uint8)

            rpn_rois = denorm_boxes(rpn_rois[0], self.config.IMAGE_SHAPE[:3])

            roi_volumes = np.power((rpn_rois[:, 3] - rpn_rois[:, 0]) * (rpn_rois[:, 4] - rpn_rois[:, 1]) * (rpn_rois[:, 5] - rpn_rois[:, 2]), 1./3.)

            box_overlaps = compute_overlaps(gt_boxes, rpn_rois)

            roi_ious = np.max(box_overlaps, axis=0)

            gt_class = np.where(roi_ious > 0.5, 1, 0).astype(np.uint8)

            (TN, FN), (FP, TP) = confusion_matrix(gt_class, mrcnn_class)

            TN_rois = np.argwhere( (1-mrcnn_class) * (1-gt_class) )[:, 0]
            FN_rois = np.argwhere( (1-mrcnn_class) * gt_class )[:, 0]
            FP_rois = np.argwhere( mrcnn_class * (1-gt_class) )[:, 0]
            TP_rois = np.argwhere( mrcnn_class * gt_class )[:, 0]

            TN_ious = roi_ious[TN_rois]
            FN_ious = roi_ious[FN_rois]
            FP_ious = roi_ious[FP_rois]
            TP_ious = roi_ious[TP_rois]

            TN_volumes = roi_volumes[TN_rois]
            FN_volumes = roi_volumes[FN_rois]
            FP_volumes = roi_volumes[FP_rois]
            TP_volumes = roi_volumes[TP_rois]

            print("Name: ", name, ", Instance number: ", gt_boxes.shape[0])
            print("True negatives: ", TN, np.mean(TN_ious), np.std(TN_ious), np.mean(TN_volumes), np.std(TN_volumes))
            print("False negatives: ", FN, np.mean(FN_ious), np.std(FN_ious), np.mean(FN_volumes), np.std(FN_volumes))
            print("False positives: ", FP, np.mean(FP_ious), np.std(FP_ious), np.mean(FP_volumes), np.std(FP_volumes))
            print("True positives: ", TP, np.mean(TP_ious), np.std(TP_ious), np.mean(TP_volumes), np.std(TP_volumes))

            

        

    def unmold_detections(self, detections, mrcnn_mask):
        """
        Reformats the detections of one image from the format of the neural
        network output to a format suitable for use in the rest of the
        application.

        detections: [N, (y1, x1, z1, y2, x2, z2, class_id, score)] in normalized coordinates
        mrcnn_mask: [N, height, width, depth, num_classes]

        Returns:
        boxes: [N, (y1, x1, z1, y2, x2, z2)] Bounding boxes in pixels
        class_ids: [N] Integer class IDs for each bounding box
        scores: [N] Float probability scores of the class_id
        masks: [height, width, depth, num_instances] Instance masks
        """
        # How many detections do we have?
        # Detections array is padded with zeros. Find the first class_id == 0.
        original_image_shape = self.config.IMAGE_SHAPE[:3]
        zero_ix = np.where(detections[:, 6] == 0)[0]
        N = zero_ix[0] if zero_ix.shape[0] > 0 else detections.shape[0]

        # Extract boxes, class_ids, scores, and class-specific masks
        boxes = detections[:N, :6]
        class_ids = detections[:N, 6].astype(np.int32)
        scores = detections[:N, 7]
        masks = mrcnn_mask[np.arange(N), ..., class_ids]

        # Convert boxes to pixel coordinates on the original image
        boxes = utils.denorm_boxes(boxes, original_image_shape)

        # Filter out detections with zero area. Happens in early training when
        # network weights are still random
        exclude_ix = np.where(
            (boxes[:, 3] - boxes[:, 0]) * (boxes[:, 4] - boxes[:, 1]) * (boxes[:, 5] - boxes[:, 2]) <= 0)[0]
        if exclude_ix.shape[0] > 0:
            boxes = np.delete(boxes, exclude_ix, axis=0)
            scores = np.delete(scores, exclude_ix, axis=0)
            masks = np.delete(masks, exclude_ix, axis=0)
            N = class_ids.shape[0]

        # Resize masks to original image size and set boundary threshold.
        segs = np.zeros((original_image_shape)).astype(np.uint16)
        for i in range(N):
            # Convert neural network mask to full size mask
            full_mask = utils.unmold_mask(masks[-i-1], boxes[-i-1], original_image_shape)
            segs = np.where(full_mask, i+1, segs)

        return boxes, scores, class_ids, segs
    
    def unmold_groundtruth(self, gt_boxes, gt_masks):
        """
        Reformats the detections of one image from the format of the neural
        network output to a format suitable for use in the rest of the
        application.

        Returns:
        segs
        """
        # How many detections do we have?
        # Detections array is padded with zeros. Find the first class_id == 0.
        original_image_shape = self.config.IMAGE_SHAPE[:3]
        N = gt_boxes.shape[0]

        # Resize masks to original image size and set boundary threshold.
        segs = np.zeros((original_image_shape)).astype(np.uint16)
        for i in range(N):
            # Convert neural network mask to full size mask
            full_mask = utils.unmold_mask(gt_masks[..., i], gt_boxes[i].astype(np.int32), original_image_shape)
            segs = np.where(full_mask, i+1, segs)

        return segs

    def get_anchors(self, image_shape):
        """Returns anchor pyramid for the given image size."""
        backbone_shapes = compute_backbone_shapes(self.config, image_shape)
        # Cache anchors and reuse if image shape is the same
        if not hasattr(self, "_anchor_cache"):
            self._anchor_cache = {}
        if not tuple(image_shape) in self._anchor_cache:
            # Generate Anchors
            a = utils.generate_pyramid_anchors(
                self.config.RPN_ANCHOR_SCALES,
                self.config.RPN_ANCHOR_RATIOS,
                backbone_shapes,
                self.config.BACKBONE_STRIDES,
                self.config.RPN_ANCHOR_STRIDE)
            # Normalize coordinates
            self._anchor_cache[tuple(image_shape)] = utils.norm_boxes(a, image_shape[:3])
        return self._anchor_cache[tuple(image_shape)]


############################################################
#  Data Formatting
############################################################


def compose_image_meta(image_id, original_image_shape, image_shape,
                       window, scale, active_class_ids):
    """Takes attributes of an image and puts them in one 1D array.

    image_id: An int ID of the image. Useful for debugging.
    original_image_shape: [H, W, D, C] before resizing or padding.
    image_shape: [H, W, D, C] after resizing and padding
    window: (y1, x1, z1, y2, x2, z2) in pixels. The area of the image where the real
            image is (excluding the padding)
    scale: The scaling factor applied to the original image (float32)
    active_class_ids: List of class_ids available in the dataset from which
        the image came. Useful if training on images from multiple datasets
        where not all classes are present in all datasets.
    """
    meta = np.array(
        [image_id] +  # size=1
        list(original_image_shape) +  # size=4
        list(image_shape) +  # size=4
        list(window) +  # size=6 (y1, x1, z1, y2, x2, z2) in image coordinates
        [scale] +  # size=1
        list(active_class_ids)  # size=num_classes
    )
    return meta #18 for num_classes = 2


def parse_image_meta(meta):
    """Parses an array that contains image attributes to its components.
    See compose_image_meta() for more details.

    meta: [batch, meta length] where meta length depends on NUM_CLASSES

    Returns a dict of the parsed values.
    """
    image_id = meta[:, 0]
    original_image_shape = meta[:, 1:5]
    image_shape = meta[:, 5:9]
    window = meta[:, 9:15]  # (y1, x1, y2, x2) window of image in in pixels
    scale = meta[:, 15]
    active_class_ids = meta[:, 16:]
    return {
        "image_id": image_id.astype(np.int32),
        "original_image_shape": original_image_shape.astype(np.int32),
        "image_shape": image_shape.astype(np.int32),
        "window": window.astype(np.int32),
        "scale": scale.astype(np.float32),
        "active_class_ids": active_class_ids.astype(np.int32),
    }


def parse_image_meta_graph(meta):
    """Parses a tensor that contains image attributes to its components.
    See compose_image_meta() for more details.

    meta: [batch, meta length] where meta length depends on NUM_CLASSES

    Returns a dict of the parsed tensors.
    """
    image_id = meta[:, 0]
    original_image_shape = meta[:, 1:5]
    image_shape = meta[:, 5:9]
    window = meta[:, 9:15]  # (y1, x1, y2, x2) window of image in in pixels
    scale = meta[:, 15]
    active_class_ids = meta[:, 16:]
    return {
        "image_id": image_id,
        "original_image_shape": original_image_shape,
        "image_shape": image_shape,
        "window": window,
        "scale": scale,
        "active_class_ids": active_class_ids,
    }


def mold_image(images, config):
    """Expects an RGB image (or array of images) and subtracts
    the mean pixel and converts it to float. Expects image
    colors in RGB order.
    """
    return images.astype(np.float32)# - config.MEAN_PIXEL


def unmold_image(normalized_images, config):
    """Takes a image normalized with mold() and returns the original."""
    return normalized_images.astype(np.uint8) #(normalized_images + config.MEAN_PIXEL)


############################################################
#  Miscellenous Graph Functions
############################################################

def trim_zeros_graph(boxes, name='trim_zeros'):
    """Often boxes are represented with matrices of shape [N, 6] and
    are padded with zeros. This removes zero boxes.

    boxes: [N, 6] matrix of boxes.
    non_zeros: [N] a 1D boolean mask identifying the rows to keep
    """
    non_zeros = tf.cast(tf.reduce_sum(tf.abs(boxes), axis=1), tf.bool)
    boxes = tf.boolean_mask(boxes, non_zeros, name=name)
    return boxes, non_zeros


def batch_pack_graph(x, counts, num_rows):
    """Picks different number of values from each row
    in x depending on the values in counts.
    """
    outputs = []
    for i in range(num_rows):
        outputs.append(x[i, :counts[i]])
    return tf.concat(outputs, axis=0)


def norm_boxes_graph(boxes, shape):
    """
    Converts boxes from pixel coordinates to normalized coordinates.

    Args:
        boxes: [..., (y1, x1, z1, y2, x2, z2)] in pixel coordinates
        shape: [..., (height, width, depth)] in pixels

    Note: In pixel coordinates (y2, x2) is outside the box. But in normalized
    coordinates it's inside the box.

    Returns:
        [..., (y1, x1, z1, y2, x2, z2)] in normalized coordinates
    """
    h, w, d = tf.split(tf.cast(shape, tf.float32), 3)
    scale = tf.concat([h, w, d, h, w, d], axis=-1) - tf.constant(1.0)
    shift = tf.constant([0., 0., 0., 1., 1., 1.])
    return tf.divide(boxes - shift, scale)


def denorm_boxes_graph(boxes, shape):
    """Converts boxes from normalized coordinates to pixel coordinates.
    boxes: [..., (y1, x1, z1, y2, x2, z2)] in normalized coordinates
    shape: [..., (height, width, depth)] in pixels

    Note: In pixel coordinates (y2, x2, z2) is outside the box. But in normalized
    coordinates it's inside the box.

    Returns:
        [..., (y1, x1, z1, y2, x2, z2)] in pixel coordinates
    """
    h, w, d = tf.split(tf.cast(shape, tf.float32), 3)
    scale = tf.concat([h, w, d, h, w, d], axis=-1) - tf.constant(1.0)
    shift = tf.constant([0., 0., 0., 1., 1., 1.])
    return tf.cast(tf.round(tf.multiply(boxes, scale) + shift), tf.int32)
