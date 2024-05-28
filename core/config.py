import numpy as np
import json


# Base Configuration Class
# Don't use this class directly. Instead, sub-class it and override
# the configurations you need to change.
# See examples in configs/

class Config(object):
    """
    Base configuration class. For custom configurations, create a
    sub-class that inherits from this one and override properties
    that need to be changed.
    """

    def __init__(
        self,
        # Data
        DATA_DIR = "data/",
        NUM_CLASSES = 2,
        IMAGE_SIZE = 256, 
        IMAGE_CHANNEL_COUNT = 1,
        MAX_GT_INSTANCES = 771,
        TARGET_RATIO = 0.2,
        USE_MINI_MASK = True,
        MINI_MASK_SHAPE = (56, 56, 56),
        RPN_BBOX_STD_DEV = [0.1, 0.1, 0.1, 0.2, 0.2, 0.2],
        BBOX_STD_DEV = [0.1, 0.1, 0.1, 0.2, 0.2, 0.2],
        EVALUATION_STEPS = 100,
        OUTPUT_DIR = "data/output/",

        # General
        MODE = "training", 

        # RPN
        BACKBONE = "resnet50",
        BACKBONE_STRIDES = [4, 8, 16, 32, 64],
        TOP_DOWN_PYRAMID_SIZE = 256,
        RPN_ANCHOR_SCALES = (32, 64, 128, 256, 512),
        RPN_ANCHOR_RATIOS = [1], 
        RPN_ANCHOR_STRIDE = 1,
        RPN_TRAIN_ANCHORS_PER_IMAGE = 256,
        RPN_NMS_THRESHOLD = 0.7, 
        PRE_NMS_LIMIT = 6000,
        POST_NMS_ROIS_TRAINING = 2000, 
        POST_NMS_ROIS_INFERENCE = 1000,

        # Head
        TRAIN_ROIS_PER_IMAGE = 200, 
        ROI_POSITIVE_RATIO = 0.33, 
        POOL_SIZE = 7, 
        MASK_POOL_SIZE = 14,
        FPN_CLASSIF_FC_LAYERS_SIZE = 1024, 
        HEAD_CONV_CHANNEL = 256,
        MASK_SHAPE = [28, 28, 28], 

        # Detection
        DETECTION_MAX_INSTANCES = 100,
        DETECTION_MIN_CONFIDENCE = 0.7, 
        DETECTION_NMS_THRESHOLD = 0.3,

        # Training
        IMAGES_PER_GPU = 1, 
        GPU_COUNT = 1, 
        LOSS_WEIGHTS = {"rpn_class_loss": 1., "rpn_bbox_loss": 1., "mrcnn_class_loss": 1., "mrcnn_bbox_loss": 1., "mrcnn_mask_loss": 1.}, 
        TRAIN_BN = False,
        LEARNING_LAYERS = "all", 
        OPTIMIZER = {"name": "SGD", "parameters": {}},
        WEIGHT_DIR = None,
        RPN_WEIGHTS = None,
        HEAD_WEIGHTS = None,
        MASK_WEIGHTS = None,
        EPOCHS = 1, 
        FROM_EPOCH = 0,
        WEIGHT_DECAY = 0.0001
    ):

        ###########################################
        ###   Data parameters
        ###########################################

        # Path to ground truth data dirs
        # In RPN targeting mode, represent the save path of head target generation
        self.DATA_DIR = DATA_DIR

        # Number of classes (including background)
        self.NUM_CLASSES = NUM_CLASSES

        # Input image size
        self.IMAGE_SIZE = IMAGE_SIZE

        # Number of color channels per image.
        self.IMAGE_CHANNEL_COUNT = IMAGE_CHANNEL_COUNT

        # Input image shape
        self.IMAGE_SHAPE = np.array([self.IMAGE_SIZE, self.IMAGE_SIZE, self.IMAGE_SIZE, self.IMAGE_CHANNEL_COUNT])

        # Maximum number of ground truth instances to use in one image
        self.MAX_GT_INSTANCES = MAX_GT_INSTANCES

        # Ratio of train/test dataset example used to generate Head targets (between 0. and 1.)
        self.TARGET_RATIO = TARGET_RATIO

        # Use of mini masks
        # Mini masks results from full masks cropping around the instance masks 
        # according to their bounding boxes and resized to a standard shape
        # If enabled, ground truth data must match the mini mask shape
        self.USE_MINI_MASK = USE_MINI_MASK
        self.MINI_MASK_SHAPE = MINI_MASK_SHAPE  # (height, width, depth) of the mini-mask

        # Bounding box refinement standard deviation for RPN and final detections.
        self.RPN_BBOX_STD_DEV = np.asarray(RPN_BBOX_STD_DEV)
        self.BBOX_STD_DEV = np.asarray(BBOX_STD_DEV)

        # Number of example used in quick evaluation between epochs (see Callbacks in models.py)
        self.EVALUATION_STEPS = EVALUATION_STEPS

        # Output path for RPN targeting or Mask inference / evaluation
        self.OUTPUT_DIR = OUTPUT_DIR


        ###########################################
        ###   General structural parameters
        ###########################################

        # Network mode
        # RPN mode either "training" or "targeting" for resp. train and head target generation
        # Head mode is "training"
        # Mask R-CNN mode either "training" or "inference" for resp. train and evaluation / prediction
        self.MODE = MODE


        ###########################################
        ###   RPN and anchor system parameters
        ###########################################

        # Backbone network architecture, "resnet50" or "resnet101" supported
        self.BACKBONE = BACKBONE

        # The strides of each layer of the FPN Pyramid. Default values based on a Resnet101 backbone.
        self.BACKBONE_STRIDES = BACKBONE_STRIDES

        # Channel number of the final top-down layers in the FPN
        self.TOP_DOWN_PYRAMID_SIZE = TOP_DOWN_PYRAMID_SIZE

        # Default anchor sizes, one for each pyramid layer (see utils.generate_pyramid_anchors
        self.RPN_ANCHOR_SCALES = RPN_ANCHOR_SCALES

        # Ratios of anchors at each cell, kept at [1] for simplicity
        # In order to play with this parameter, must adapt utils.generate_pyramid_anchors
        # We developed one, ask in Git issue
        self.RPN_ANCHOR_RATIOS = RPN_ANCHOR_RATIOS

        # Anchor stride
        # If 1 then anchors are created for each cell in the backbone feature map.
        # If 2, then anchors are created for every other cell, and so on.
        self.RPN_ANCHOR_STRIDE = RPN_ANCHOR_STRIDE

        # How many anchors per image to use for RPN training
        self.RPN_TRAIN_ANCHORS_PER_IMAGE = RPN_TRAIN_ANCHORS_PER_IMAGE

        # Non-max suppression threshold to filter RPN proposals.
        # You can increase this during training to generate more propsals.
        self.RPN_NMS_THRESHOLD = RPN_NMS_THRESHOLD

        # ROIs kept after tf.nn.top_k and before non-maximum suppression
        self.PRE_NMS_LIMIT = PRE_NMS_LIMIT

        # ROIs kept after non-maximum suppression (training and inference)
        self.POST_NMS_ROIS_TRAINING = POST_NMS_ROIS_TRAINING
        self.POST_NMS_ROIS_INFERENCE = POST_NMS_ROIS_INFERENCE

        # Compute the number of anchors, useful in debugging
        self.ANCHOR_NB = int( (self.IMAGE_SHAPE[0] / self.BACKBONE_STRIDES[0])**3 + \
                            (self.IMAGE_SHAPE[0] / self.BACKBONE_STRIDES[1])**3 + \
                            (self.IMAGE_SHAPE[0] / self.BACKBONE_STRIDES[2])**3 + \
                            (self.IMAGE_SHAPE[0] / self.BACKBONE_STRIDES[3])**3 + \
                            (self.IMAGE_SHAPE[0] / self.BACKBONE_STRIDES[4])**3 )
        

        ###########################################
        ###   Head parameters
        ###########################################

        # Number of ROIs per image to feed to classifier/mask heads
        # You can increase the number of proposals by adjusting the RPN NMS threshold.
        self.TRAIN_ROIS_PER_IMAGE = TRAIN_ROIS_PER_IMAGE

        # Percent of positive ROIs used to train classifier/mask heads
        # Between 0. and 1.
        # 0.3 means that positive RoIs will represent at most 0.3 of the total RoI number.
        self.ROI_POSITIVE_RATIO = ROI_POSITIVE_RATIO

        # Size of RoIs pooled from feature maps
        self.POOL_SIZE = POOL_SIZE
        self.MASK_POOL_SIZE = MASK_POOL_SIZE

        # Size of the fully-connected layers in the classification graph
        self.FPN_CLASSIF_FC_LAYERS_SIZE = FPN_CLASSIF_FC_LAYERS_SIZE

        # Feature channel number used in the segmentation module
        self.HEAD_CONV_CHANNEL = HEAD_CONV_CHANNEL
        
        # Shape of output mask
        # To change this you also need to change the neural network mask branch
        self.MASK_SHAPE = MASK_SHAPE


        ###########################################
        ###   Detection parameters
        ###########################################

        # Max number of final detections
        self.DETECTION_MAX_INSTANCES = DETECTION_MAX_INSTANCES

        # Minimum probability value to accept a detected instance
        # ROIs below this threshold are skipped
        self.DETECTION_MIN_CONFIDENCE = DETECTION_MIN_CONFIDENCE

        # Non-maximum suppression threshold for detection
        self.DETECTION_NMS_THRESHOLD = DETECTION_NMS_THRESHOLD


        ###########################################
        ###   Training parameters
        ###########################################

        # Number of images to train with on each GPU
        self.IMAGES_PER_GPU = IMAGES_PER_GPU

        # Number of GPUs to use.
        # When using only a CPU, this needs to be set to 1.
        self.GPU_COUNT = GPU_COUNT

        # Effective batch size
        self.BATCH_SIZE = self.IMAGES_PER_GPU * self.GPU_COUNT

        # Image meta data length (see models.compose_image_meta for details)
        self.IMAGE_META_SIZE = 1 + 4 + 4 + 6 + 1 + self.NUM_CLASSES
 
        # Loss weights for more precise optimization.
        self.LOSS_WEIGHTS = LOSS_WEIGHTS

        # Train or freeze batch normalization layers
        #     None: Train BN layers. This is the normal mode
        #     False: Freeze BN layers. Good when using a small batch size
        #     True: (don't use). Set layer in training mode even when predicting
        self.TRAIN_BN = TRAIN_BN  # Defaulting to False since batch size is often small

        # Layers to train
        # Only for Mask R-CNN model (see MaskRCNN.train method for details)
        # "rpn", "head" or "all" supported
        self.LEARNING_LAYERS = LEARNING_LAYERS

        # Optimizer
        # "SGD" and "Adadelta" supported
        self.OPTIMIZER = OPTIMIZER

        # Weight save directory
        self.WEIGHT_DIR = WEIGHT_DIR

        # Pathes for RPN, Head or Mask R-CNN weights to load before training if necessary
        self.RPN_WEIGHTS = RPN_WEIGHTS
        self.HEAD_WEIGHTS = HEAD_WEIGHTS
        self.MASK_WEIGHTS = MASK_WEIGHTS

        # Number of epochs to train over
        self.EPOCHS = EPOCHS

        # Use to track the training
        # Must be 0 in case of new config/training
        # Must correspond to the epoch number of RPN_WEIGHTS or HEAD_WEIGHTS or MASK_WEIGHTS
        self.FROM_EPOCH = FROM_EPOCH

        # Weight decay regularization
        self.WEIGHT_DECAY = WEIGHT_DECAY
    
    def display(self):
        """
        Display Configuration values.
        """
        print("\nConfigurations:")
        for a in dir(self):
            if not a.startswith("__") and not callable(getattr(self, a)):
                print("{:30} {}".format(a, getattr(self, a)))
        print("\n")


def load_config(config_path):
    with open(config_path) as config_file:
        config_dict = json.load(config_file)
    config = Config(**config_dict)
    return config
