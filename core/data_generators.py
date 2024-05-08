"""
3D Mask R-CNN

Based on the 2D implementation by Matterport, Inc,
the update to TensorFlow 2 written by Ahmed Gad and
the world4jason fork that replaced the old data generator
for the DataGenerator from keras.utils.Sequence.

https://github.com/matterport/Mask_RCNN
https://github.com/ahmedfgad/Mask-RCNN-TF2
https://github.com/matterport/Mask_RCNN/pull/1611/files

This 3D implementation was written by Gabriel David (PhD).

Licensed under the MIT License (see Matterport_LICENSE for details)
"""

import math
import bz2
import _pickle as cPickle
import numpy as np
import pandas as pd
from skimage.io import imread
import keras
from core import utils


############################################################
#  Data Generator
############################################################

class HeadGenerator(keras.utils.Sequence):

    def __init__(self, dataset, config, shuffle=True):

        self.image_ids = np.copy(dataset.image_ids)
        self.dataset = dataset
        self.config = config
        self.shuffle = shuffle
        self.batch_size = self.config.BATCH_SIZE

    def __len__(self):
        return int(np.ceil(len(self.image_ids) / float(self.batch_size)))

    def __getitem__(self, idx):
        return self.data_generator(self.image_ids[idx * self.batch_size:(idx + 1) * self.batch_size])

    def data_generator(self, image_ids):
        image_id = image_ids[0]

        rois_aligned, mask_aligned, image_meta, target_class_ids, target_bbox, target_mask = self.load_image_gt(image_id)

        inputs = [rois_aligned[np.newaxis], mask_aligned[np.newaxis], image_meta[np.newaxis],
                    target_class_ids[np.newaxis], target_bbox[np.newaxis], target_mask[np.newaxis, ..., np.newaxis]]
        outputs = []

        return inputs, outputs

    def load_image_gt(self, image_id):
        rois_aligned, mask_aligned, target_class_ids, target_bbox, target_mask = self.dataset.load_data(image_id)

        active_class_ids = np.zeros([self.dataset.num_classes], dtype=np.int32)
        source_class_ids = self.dataset.source_class_ids[self.dataset.image_info[image_id]["source"]]
        active_class_ids[source_class_ids] = 1

        image_meta = compose_image_meta(image_id, tuple(self.config.IMAGE_SHAPE), tuple(self.config.IMAGE_SHAPE), (0, 0, 0, *self.config.IMAGE_SHAPE[:-1]), 1,
                                        active_class_ids)
        
        return rois_aligned, mask_aligned, image_meta, target_class_ids, target_bbox, target_mask

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.image_ids)


class RPNGenerator(keras.utils.Sequence):

    def __init__(self, dataset, config, shuffle=True):

        self.image_ids = np.copy(dataset.image_ids)
        self.dataset = dataset
        self.config = config
        self.shuffle = shuffle
        self.batch_size = self.config.BATCH_SIZE
        backbone_shapes = compute_backbone_shapes(config, config.IMAGE_SHAPE)
        self.anchors = utils.generate_pyramid_anchors(config.RPN_ANCHOR_SCALES, config.RPN_ANCHOR_RATIOS, backbone_shapes,
                                   config.BACKBONE_STRIDES, config.RPN_ANCHOR_STRIDE)

    def __len__(self):
        return int(np.ceil(len(self.image_ids) / float(self.batch_size)))

    def __getitem__(self, idx):
        return self.data_generator(self.image_ids[idx * self.batch_size:(idx + 1) * self.batch_size])

    def data_generator(self, image_ids):
        b = 0
        while b < self.batch_size:
            # Get GT bounding boxes and masks for image.
            image_id = image_ids[b]

            if self.config.MODE == "targeting":
                image, image_meta, gt_class_ids, gt_boxes, gt_masks = self.load_image_gt(image_id)

                # Init batch arrays
                if b == 0:
                    batch_image_meta = np.zeros(
                        (self.batch_size,) + image_meta.shape, dtype=image_meta.dtype)
                    batch_images = np.zeros(
                        (self.batch_size,) + image.shape, dtype=np.float32)
                    batch_gt_class_ids = np.zeros(
                        (self.batch_size, self.config.MAX_GT_INSTANCES), dtype=np.int32)
                    batch_gt_boxes = np.zeros(
                        (self.batch_size, self.config.MAX_GT_INSTANCES, 6), dtype=np.int32)
                    batch_gt_masks = np.zeros(
                        (self.batch_size, gt_masks.shape[0], gt_masks.shape[1], gt_masks.shape[2],
                         self.config.MAX_GT_INSTANCES), dtype=gt_masks.dtype)

                batch_image_meta[b] = image_meta
                batch_images[b] = image.astype(np.float32)
                batch_gt_class_ids[b, :gt_class_ids.shape[0]] = gt_class_ids
                batch_gt_boxes[b, :gt_boxes.shape[0]] = gt_boxes
                batch_gt_masks[b, ..., :gt_masks.shape[-1]] = gt_masks
                b += 1

                if b >= self.batch_size:
                    inputs = [batch_images, batch_image_meta,  batch_gt_class_ids, batch_gt_boxes, batch_gt_masks]
                    outputs = []

                    return inputs, outputs
            else:
                image, _, rpn_match, rpn_bbox = self.load_image_gt(image_id)

                # Init batch arrays
                if b == 0:
                    batch_rpn_match = np.zeros([self.batch_size, self.config.ANCHOR_NB, 1], dtype=rpn_match.dtype)
                    batch_rpn_bbox = np.zeros([self.batch_size, self.config.RPN_TRAIN_ANCHORS_PER_IMAGE, 6],
                                                dtype=rpn_bbox.dtype)
                    batch_images = np.zeros((self.batch_size,) + image.shape, dtype=np.float32)

                # Add to batch
                batch_rpn_match[b] = rpn_match[:, np.newaxis]
                batch_rpn_bbox[b] = rpn_bbox
                batch_images[b] = image.astype(np.float32)
                b += 1

                # Batch full?
                if b >= self.batch_size:
                    inputs = [batch_images, batch_rpn_match, batch_rpn_bbox]
                    outputs = []

                    return inputs, outputs

    def load_image_gt(self, image_id):
        # Load image and mask
        image = self.dataset.load_image(image_id)
        if self.config.MODE == "targeting":
            boxes, class_ids, masks, _, _ = self.dataset.load_data(image_id)
            active_class_ids = np.zeros([self.dataset.num_classes], dtype=np.int32)
            source_class_ids = self.dataset.source_class_ids[self.dataset.image_info[image_id]["source"]]
            active_class_ids[source_class_ids] = 1

            image_meta = compose_image_meta(image_id, tuple(self.config.IMAGE_SHAPE), tuple(self.config.IMAGE_SHAPE), 
                                            (0, 0, 0, *self.config.IMAGE_SHAPE[:-1]), 1, active_class_ids)
            return image, image_meta, class_ids, boxes, masks
        else:
            boxes, _, _, rpn_match, rpn_bbox = self.dataset.load_data(image_id, gt_needed=False)
            return image, boxes, rpn_match, rpn_bbox

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.image_ids)


class MrcnnGenerator(keras.utils.Sequence):

    def __init__(self, dataset, config, shuffle=True, batch_size=1, training=True):

        self.image_ids = np.copy(dataset.image_ids)
        np.random.shuffle(self.image_ids)
        self.dataset = dataset
        self.config = config
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.training = training
        self.backbone_shapes = compute_backbone_shapes(config, config.IMAGE_SHAPE)
        self.anchors = utils.norm_boxes(utils.generate_pyramid_anchors(config.RPN_ANCHOR_SCALES,
                                                                        config.RPN_ANCHOR_RATIOS,
                                                                        self.backbone_shapes,
                                                                        config.BACKBONE_STRIDES,
                                                                        config.RPN_ANCHOR_STRIDE), 
                                                                        tuple(config.IMAGE_SHAPE[:-1]))

    def __len__(self):
        return int(np.ceil(len(self.image_ids) / float(self.batch_size)))

    def __getitem__(self, idx):
        return self.data_generator(self.image_ids[idx * self.batch_size:(idx + 1) * self.batch_size])

    def data_generator(self, image_ids):
        b = 0
        while b < self.batch_size:
            # Get GT bounding boxes and masks for image.
            image_id = image_ids[b]
            if self.training:
                image, image_meta, gt_boxes, gt_class_ids, gt_masks = self.load_image_gt(image_id)
                rpn_match, rpn_bbox = build_rpn_targets(self.anchors, gt_class_ids, gt_boxes, self.config)
                # Init batch arrays
                if b == 0:
                    batch_image_meta = np.zeros(
                        (self.batch_size,) + image_meta.shape, dtype=image_meta.dtype)
                    batch_images = np.zeros(
                        (self.batch_size,) + image.shape, dtype=np.float32)
                    batch_gt_class_ids = np.zeros(
                        (self.batch_size, self.config.MAX_GT_INSTANCES), dtype=np.int32)
                    batch_gt_boxes = np.zeros(
                        (self.batch_size, self.config.MAX_GT_INSTANCES, 6), dtype=np.int32)
                    batch_gt_masks = np.zeros(
                        (self.batch_size, gt_masks.shape[0], gt_masks.shape[1], gt_masks.shape[2],
                         self.config.MAX_GT_INSTANCES), dtype=gt_masks.dtype)
                    batch_rpn_match = np.zeros([self.batch_size, self.config.ANCHOR_NB, 1], dtype=rpn_match.dtype)
                    batch_rpn_bbox = np.zeros([self.batch_size, self.config.RPN_TRAIN_ANCHORS_PER_IMAGE, 6],
                                              dtype=rpn_bbox.dtype)

                batch_image_meta[b] = image_meta
                batch_images[b] = image.astype(np.float32)
                batch_gt_class_ids[b, :gt_class_ids.shape[0]] = gt_class_ids
                batch_gt_boxes[b, :gt_boxes.shape[0]] = gt_boxes
                batch_gt_masks[b, ..., :gt_masks.shape[-1]] = gt_masks
                batch_rpn_match[b] = rpn_match[:, np.newaxis]
                batch_rpn_bbox[b] = rpn_bbox
                b += 1

                if b >= self.batch_size:
                    inputs = [batch_images, batch_image_meta,  batch_gt_class_ids, batch_gt_boxes, batch_gt_masks,
                              batch_rpn_match, batch_rpn_bbox]
                    outputs = []

                    return inputs, outputs
            else:
                image, image_meta = self.load_image_gt(image_id)

                # Init batch arrays
                if b == 0:
                    batch_image_meta = np.zeros(
                        (self.batch_size,) + image_meta.shape, dtype=image_meta.dtype)
                    batch_images = np.zeros(
                        (self.batch_size,) + image.shape, dtype=np.float32)
                    batch_anchors = self.anchors[np.newaxis]

                batch_image_meta[b] = image_meta
                batch_images[b] = image.astype(np.float32)
                b += 1

                if b >= self.batch_size:
                    inputs = [batch_images, batch_image_meta, batch_anchors]
                    outputs = []

                    return inputs, outputs

    def load_image_gt(self, image_id):
        # Load image and mask
        image = self.dataset.load_image(image_id)
        active_class_ids = np.zeros([self.dataset.num_classes], dtype=np.int32)
        source_class_ids = self.dataset.source_class_ids[self.dataset.image_info[image_id]["source"]]
        active_class_ids[source_class_ids] = 1
        image_meta = compose_image_meta(image_id, tuple(self.config.IMAGE_SHAPE), tuple(self.config.IMAGE_SHAPE), 
                                        (0, 0, 0, *self.config.IMAGE_SHAPE[:-1]), 1, active_class_ids)
        if self.training:
            boxes, class_ids, masks = self.dataset.load_data(image_id)
            return image, image_meta, boxes, class_ids, masks
        else:
            return image, image_meta

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.image_ids)

    def get_input_prediction(self, image_id):
        image_path = self.dataset.image_info[image_id]["path"]
        name = image_path.split("/")[-1]
        image_meta = compose_image_meta(image_id, tuple(self.config.IMAGE_SHAPE), tuple(self.config.IMAGE_SHAPE), 
                                        (0, 0, 0, *self.config.IMAGE_SHAPE[:-1]), 1, np.array([1 for _ in range(self.config.NUM_CLASSES)], dtype=np.int32))
        image = imread(image_path)
        image = 2 * (image / 255) - 1
        image = image[..., np.newaxis]
        batch_image_meta = np.zeros(
            (self.batch_size,) + image_meta.shape, dtype=image_meta.dtype)
        batch_images = np.zeros(
            (self.batch_size,) + image.shape, dtype=np.float32)
        batch_anchors = self.anchors[np.newaxis]
        batch_image_meta[0] = image_meta
        batch_images[0] = image.astype(np.float32)
        inputs = [batch_images, batch_image_meta, batch_anchors]
        return name, inputs


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
#  Dataset Objects
############################################################

class Dataset(object):
    """The base class for dataset classes.
    To use it, create a new class that adds functions specific to the dataset
    you want to use. For example:

    class CatsAndDogsDataset(Dataset):
        def load_cats_and_dogs(self):
            ...
        def load_mask(self, image_id):
            ...
        def image_reference(self, image_id):
            ...

    See COCODataset and ShapesDataset as examples.
    """

    def __init__(self, class_map=None):
        self._image_ids = []
        self.image_info = []
        # Background is always the first class
        self.class_info = [{"source": "", "id": 0, "name": "BG"}]
        self.source_class_ids = {}

    def add_class(self, source, class_id, class_name):
        assert "." not in source, "Source name cannot contain a dot"
        # Does the class exist already?
        for info in self.class_info:
            if info['source'] == source and info["id"] == class_id:
                # source.class_id combination already available, skip
                return
        # Add the class
        self.class_info.append({
            "source": source,
            "id": class_id,
            "name": class_name,
        })

    def add_image(self, source, image_id, path, **kwargs):
        image_info = {
            "id": image_id,
            "source": source,
            "path": path,
        }
        image_info.update(kwargs)
        self.image_info.append(image_info)

    def prepare(self, class_map=None):
        """Prepares the Dataset class for use.

        TODO: class map is not supported yet. When done, it should handle mapping
              classes from different datasets to the same class ID.
        """

        def clean_name(name):
            """Returns a shorter version of object names for cleaner display."""
            return ",".join(name.split(",")[:1])

        # Build (or rebuild) everything else from the info dicts.
        self.num_classes = len(self.class_info)
        self.class_ids = np.arange(self.num_classes)
        self.class_names = [clean_name(c["name"]) for c in self.class_info]
        self.num_images = len(self.image_info)
        self._image_ids = np.arange(self.num_images)

        # Mapping from source class and image IDs to internal IDs
        self.class_from_source_map = {"{}.{}".format(info['source'], info['id']): id
                                      for info, id in zip(self.class_info, self.class_ids)}
        self.image_from_source_map = {"{}.{}".format(info['source'], info['id']): id
                                      for info, id in zip(self.image_info, self.image_ids)}

        # Map sources to class_ids they support
        self.sources = list(set([i['source'] for i in self.class_info]))
        self.source_class_ids = {}
        # Loop over datasets
        for source in self.sources:
            self.source_class_ids[source] = []
            # Find classes that belong to this dataset
            for i, info in enumerate(self.class_info):
                # Include BG class in all datasets
                if i == 0 or source == info['source']:
                    self.source_class_ids[source].append(i)

    def map_source_class_id(self, source_class_id):
        """Takes a source class ID and returns the int class ID assigned to it.

        For example:
        dataset.map_source_class_id("coco.12") -> 23
        """
        return self.class_from_source_map[source_class_id]

    def get_source_class_id(self, class_id, source):
        """Map an internal class ID to the corresponding class ID in the source dataset."""
        info = self.class_info[class_id]
        assert info['source'] == source
        return info['id']

    @property
    def image_ids(self):
        return self._image_ids

    def source_image_link(self, image_id):
        """Returns the path or URL to the image.
        Override this to return a URL to the image if it's available online for easy
        debugging.
        """
        return self.image_info[image_id]["path"]


class EmbryoDataset(Dataset):

    def load_dataset(self, data_dir, is_train=True):
        '''
        An image is defined in the dataset by:
        # image_id, an image id [type: str of size 7]
        # path, the path of its origin 3D image [type: str]
        # seg_path, the path of the segmented 3D image [type: str]
        # ann_time_step, the time step of the origin 3D image (for validation test) [type: str of size 3]
        # ann_section_type, the type of section of the origin 3D image (x, y or z) [type: str of form 'x-section']
        # ann_section_id, the index of the section in regard to the origin image [type: str of size 3]
        # ann_mean = the mean of the origin 3D image pixel values [type: float]
        # ann_std = the standard deviation of the origin 3D image pixel values
        '''
        self.add_class("dataset", 1, "embryo")

        if is_train:
            td = pd.read_csv(f"{data_dir}datasets/train.csv", header=[0])
            for i in range(len(td)):
                img_path = td["images"][i]
                bx_path = td["boxes"][i]
                m_path = td["minimasks"][i]
                rm_path = td["rpn_match"][i]
                rb_path = td["rpn_bbox"][i]
                self.add_image('dataset', image_id=i, path=img_path, bx_path=bx_path, m_path=m_path, rm_path=rm_path,
                               rb_path=rb_path)
            print('Training dataset is loaded.')
        else:
            td = pd.read_csv(f"{data_dir}datasets/test.csv", header=[0])
            for i in range(len(td)):
                img_path = td["images"][i]
                bx_path = td["boxes"][i]
                m_path = td["minimasks"][i]
                rm_path = td["rpn_match"][i]
                rb_path = td["rpn_bbox"][i]
                self.add_image('dataset', image_id=i, path=img_path, bx_path=bx_path, m_path=m_path, rm_path=rm_path,
                               rb_path=rb_path)
            print('Validation dataset is loaded.')

    def load_image(self, image_id):
        """Load the specified image and return a [H,W,3] Numpy array.
        """
        # Load image
        info = self.image_info[image_id]
        image = imread(info["path"])
        image = 2 * (image / 255) - 1
        image = image[..., np.newaxis]
        return image

    def load_data(self, image_id, gt_needed=True):

        info = self.image_info[image_id]
        boxes = np.loadtxt(info["bx_path"])
        
        if gt_needed:

            masks = bz2.BZ2File(info["m_path"], 'rb')
            masks = cPickle.load(masks)
            class_ids = np.ones(boxes.shape[0])
            rpn_match = None
            rpn_bbox = None

        else:

            masks = None
            class_ids = None
            rpn_match = np.load(info["rm_path"])
            rpn_bbox = np.load(info["rb_path"])

        return boxes, class_ids, masks, rpn_match, rpn_bbox


class EmbryoHeadDataset(Dataset):

    def load_dataset(self, data_dir, is_train=True):
        '''
        An image is defined in the dataset by:
        # image_id, an image id [type: str of size 7]
        # path, the path of its origin 3D image [type: str]
        # seg_path, the path of the segmented 3D image [type: str]
        # ann_time_step, the time step of the origin 3D image (for validation test) [type: str of size 3]
        # ann_section_type, the type of section of the origin 3D image (x, y or z) [type: str of form 'x-section']
        # ann_section_id, the index of the section in regard to the origin image [type: str of size 3]
        # ann_mean = the mean of the origin 3D image pixel values [type: float]
        # ann_std = the standard deviation of the origin 3D image pixel values
        '''
        self.add_class("dataset", 1, "sphere")
        self.add_class("dataset", 2, "cube")
        self.add_class("dataset", 3, "pyramid")

        if is_train:
            td = pd.read_csv(f"{data_dir}datasets/train.csv", header=[0])
            for i in range(len(td)):
                r_path = td["rois"][i]
                ra_path = td["rois_aligned"][i]
                ma_path = td["mask_aligned"][i]
                tci_path = td["target_class_ids"][i]
                tb_path = td["target_bbox"][i]
                tm_path = td["target_mask"][i]
                self.add_image('dataset', image_id=i, path=r_path, ra_path=ra_path, ma_path=ma_path, tci_path=tci_path,
                               tb_path=tb_path, tm_path=tm_path)
            print('Training dataset is loaded.')
        else:
            td = pd.read_csv(f"{data_dir}datasets/test.csv", header=[0])
            for i in range(len(td)):
                r_path = td["rois"][i]
                ra_path = td["rois_aligned"][i]
                ma_path = td["mask_aligned"][i]
                tci_path = td["target_class_ids"][i]
                tb_path = td["target_bbox"][i]
                tm_path = td["target_mask"][i]
                self.add_image('dataset', image_id=i, path=r_path, ra_path=ra_path, ma_path=ma_path, tci_path=tci_path,
                               tb_path=tb_path, tm_path=tm_path)
            print('Validation dataset is loaded.')

    def load_data(self, image_id):
        info = self.image_info[image_id]
        rois_aligned = np.load(info["ra_path"])
        mask_aligned = np.load(info["ma_path"])
        target_class_ids = np.load(info["tci_path"])
        target_bbox = np.load(info["tb_path"])
        target_mask = np.load(info["tm_path"])
        return rois_aligned, mask_aligned, target_class_ids, target_bbox, target_mask



############################################################
#  Target Generators
############################################################


def build_rpn_targets(anchors, gt_class_ids, gt_boxes, config):
    """
    Given the anchors and GT boxes, compute overlaps and identify positive
    anchors and deltas to refine them to match their corresponding GT boxes.

    anchors: [num_anchors, (y1, x1, z1, y2, x2, z2)]
    gt_class_ids: [num_gt_boxes] Integer class IDs.
    gt_boxes: [num_gt_boxes, (y1, x1, z1, y2, x2, z2)]

    Returns:
    rpn_match: [N] (int32) matches between anchors and GT boxes.
               1 = positive anchor, -1 = negative anchor, 0 = neutral
    rpn_bbox: [N, (dy, dx, dz, log(dh), log(dw), log(d))] Anchor bbox deltas.
    """
    # RPN Match: 1 = positive anchor, -1 = negative anchor, 0 = neutral
    rpn_match = np.zeros([anchors.shape[0]], dtype=np.int32)
    # RPN bounding boxes: [max anchors per image, (dy, dx, log(dh), log(dw))]
    rpn_bbox = np.zeros((config.RPN_TRAIN_ANCHORS_PER_IMAGE, 6))

    # Handle COCO crowds
    # A crowd box in COCO is a bounding box around several instances. Exclude
    # them from training. A crowd box is given a negative class ID.
    crowd_ix = np.where(gt_class_ids < 0)[0]
    if crowd_ix.shape[0] > 0:
        # Filter out crowds from ground truth class IDs and boxes
        non_crowd_ix = np.where(gt_class_ids > 0)[0]
        crowd_boxes = gt_boxes[crowd_ix]
        gt_class_ids = gt_class_ids[non_crowd_ix]
        gt_boxes = gt_boxes[non_crowd_ix]
        # Compute overlaps with crowd boxes [anchors, crowds]
        crowd_overlaps = utils.compute_overlaps(anchors, crowd_boxes)
        crowd_iou_max = np.amax(crowd_overlaps, axis=1)
        no_crowd_bool = (crowd_iou_max < 0.001)
    else:
        # All anchors don't intersect a crowd
        no_crowd_bool = np.ones([anchors.shape[0]], dtype=bool)

    # Compute overlaps [num_anchors, num_gt_boxes]
    overlaps = utils.compute_overlaps(anchors, gt_boxes)

    # Match anchors to GT Boxes
    # If an anchor overlaps a GT box with IoU >= 0.7 then it's positive.
    # If an anchor overlaps a GT box with IoU < 0.3 then it's negative.
    # Neutral anchors are those that don't match the conditions above,
    # and they don't influence the loss function.
    # However, don't keep any GT box unmatched (rare, but happens). Instead,
    # match it to the closest anchor (even if its max IoU is < 0.3).
    #
    # 1. Set negative anchors first. They get overwritten below if a GT box is
    # matched to them. Skip boxes in crowd areas.
    anchor_iou_argmax = np.argmax(overlaps, axis=1)
    anchor_iou_max = overlaps[np.arange(overlaps.shape[0]), anchor_iou_argmax]
    rpn_match[(anchor_iou_max < 0.3) & (no_crowd_bool)] = -1
    # 2. Set an anchor for each GT box (regardless of IoU value).
    # If multiple anchors have the same IoU match all of them
    gt_iou_argmax = np.argwhere(overlaps == np.max(overlaps, axis=0))[:, 0]
    rpn_match[gt_iou_argmax] = 1
    # 3. Set anchors with high overlap as positive.
    rpn_match[anchor_iou_max >= 0.7] = 1

    # Subsample to balance positive and negative anchors
    # Don't let positives be more than half the anchors
    ids = np.where(rpn_match == 1)[0]
    extra = len(ids) - (config.RPN_TRAIN_ANCHORS_PER_IMAGE // 2)
    if extra > 0:
        # Reset the extra ones to neutral
        ids = np.random.choice(ids, extra, replace=False)
        rpn_match[ids] = 0
    # Same for negative proposals
    ids = np.where(rpn_match == -1)[0]
    extra = len(ids) - (config.RPN_TRAIN_ANCHORS_PER_IMAGE -
                        np.sum(rpn_match == 1))
    if extra > 0:
        # Rest the extra ones to neutral
        ids = np.random.choice(ids, extra, replace=False)
        rpn_match[ids] = 0

    # For positive anchors, compute shift and scale needed to transform them
    # to match the corresponding GT boxes.
    ids = np.where(rpn_match == 1)[0]
    ix = 0  # index into rpn_bbox
    # TODO: use box_refinement() rather than duplicating the code here
    for i, a in zip(ids, anchors[ids]):
        # Closest gt box (it might have IoU < 0.7)
        gt = gt_boxes[anchor_iou_argmax[i]]

        # Convert coordinates to center plus width/height.
        # GT Box
        gt_h = gt[3] - gt[0]
        gt_w = gt[4] - gt[1]
        gt_d = gt[5] - gt[2]
        gt_center_y = gt[0] + 0.5 * gt_h
        gt_center_x = gt[1] + 0.5 * gt_w
        gt_center_z = gt[2] + 0.5 * gt_d
        # Anchor
        a_h = a[3] - a[0]
        a_w = a[4] - a[1]
        a_d = a[5] - a[2]
        a_center_y = a[0] + 0.5 * a_h
        a_center_x = a[1] + 0.5 * a_w
        a_center_z = a[2] + 0.5 * a_d

        # Compute the bbox refinement that the RPN should predict.
        rpn_bbox[ix] = [
            (gt_center_y - a_center_y) / a_h,
            (gt_center_x - a_center_x) / a_w,
            (gt_center_z - a_center_z) / a_d,
            np.log(gt_h / a_h),
            np.log(gt_w / a_w),
            np.log(gt_d / a_d)
        ]
        # Normalize
        rpn_bbox[ix] /= config.RPN_BBOX_STD_DEV
        ix += 1

    return rpn_match, rpn_bbox
