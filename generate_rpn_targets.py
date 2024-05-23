import os
import threading
import numpy as np
import pandas as pd
from tqdm import tqdm

from core.utils import generate_pyramid_anchors
from core.data_generators import build_rpn_targets, compute_backbone_shapes
from core.config import load_config

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


config_path = "configs/rpn/embryo_rpn_config.json"
config = load_config(config_path)

backbone_shapes = compute_backbone_shapes(config, config.IMAGE_SHAPE)
anchors = generate_pyramid_anchors(config.RPN_ANCHOR_SCALES, config.RPN_ANCHOR_RATIOS, backbone_shapes,
                                   config.BACKBONE_STRIDES, config.RPN_ANCHOR_STRIDE)


data_path = "/data/icar/gdavid/Embryo3D/"
os.makedirs(f"{data_path}rpn_match/", exist_ok=True)
os.makedirs(f"{data_path}rpn_bbox/", exist_ok=True)

df_train = pd.read_csv(f"{config.DATA_DIR}datasets/train.csv", header=[0])
df_valid = pd.read_csv(f"{config.DATA_DIR}datasets/valid.csv", header=[0])
df_test = pd.read_csv(f"{config.DATA_DIR}datasets/test.csv", header=[0])
df = pd.concat([df_train, df_valid, df_test]).reset_index()

print(df)


def generate_target(path):

    dataset = pd.read_csv(path, header=[0])

    for i in tqdm(range(len(dataset))):

        box_path = dataset["boxes"][i]
        match_path = dataset["rpn_match"][i]
        bbox_path = dataset["rpn_bbox"][i]

        if not os.path.isfile(match_path):
                
            gt_boxes = np.loadtxt(box_path)

            gt_class_ids = np.ones(gt_boxes.shape[0])

            rpn_match, rpn_bbox = build_rpn_targets(anchors, gt_class_ids, gt_boxes, config)

            np.save(match_path, rpn_match)
            np.save(bbox_path, rpn_bbox)


def generate_data(full_dataset, thread_nb):

    dfs = np.array_split(full_dataset, thread_nb)

    pathes = [f"./frac{n}.csv" for n in range(thread_nb)]

    for n in range(thread_nb):
        dfs[n].to_csv(pathes[n], index=None)

    for i in range(thread_nb):
        x = threading.Thread(target=generate_target, args=(pathes[i], ))
        x.start()


generate_data(df, 5)
