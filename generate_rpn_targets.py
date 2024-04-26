from skimage.io import imread
import numpy as np
import models.rpn
import ast
import os
from models.utils import generate_pyramid_anchors, compute_overlaps, denorm_boxes
from models.data_generator import build_rpn_targets, compute_backbone_shapes
from models.config import load_config
import pandas as pd
import models.config
from tqdm import tqdm
import threading
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

config_path = "configs/rpn/rpn_config5.dat"
config = load_config(config_path)

backbone_shapes = compute_backbone_shapes(config, config.IMAGE_SHAPE)
anchors = generate_pyramid_anchors(config.RPN_ANCHOR_SCALES, config.RPN_ANCHOR_RATIOS, backbone_shapes,
                                   config.BACKBONE_STRIDES, config.RPN_ANCHOR_STRIDE)

data_path = "/data/icar/gdavid/Embryo3D/"
os.makedirs(f"{data_path}rpn_match_bis/", exist_ok=True)
os.makedirs(f"{data_path}rpn_bbox_bis/", exist_ok=True)
df_train = pd.read_csv(f"{config.DATASET_DIR}train_head.csv", header=[0, 1])
df_test = pd.read_csv(f"{config.DATASET_DIR}test.csv", header=[0, 1])
df = pd.concat([df_train, df_test])


def generate_target(path):
    dataset = pd.read_csv(path, header=[0, 1])
    for i in tqdm(range(len(dataset))):
        box_path = dataset[("Outputs", "boxes")][i]
        name = box_path.split("/")[-1].split(".")[0]
        match_path = f"{data_path}rpn_match_bis/{name}.npy"
        bbox_path = f"{data_path}rpn_bbox_bis/{name}.npy"
        if not os.path.isfile(match_path):
            if "/data/icar/" in box_path:
                gt_boxes = np.loadtxt(box_path)
            else:
                gt_boxes = np.loadtxt(box_path.replace("/data/", "/data/icar/"))
            gt_class_ids = np.ones(gt_boxes.shape[0])
            rpn_match, rpn_bbox = build_rpn_targets(anchors, gt_class_ids, gt_boxes, config)
            np.save(match_path, rpn_match)
            np.save(bbox_path, rpn_bbox)


def generate_unique_target(name, gt_boxes_path):
    gt_boxes = np.loadtxt(gt_boxes_path)
    gt_class_ids = np.ones(gt_boxes.shape[0])
    rpn_match, rpn_bbox = build_rpn_targets(anchors, gt_class_ids, gt_boxes, config)
    match_path = f"{data_path}rpn_match/{name}.npy"
    bbox_path = f"{data_path}rpn_bbox/{name}.npy"
    np.save(match_path, rpn_match)
    np.save(bbox_path, rpn_bbox)



def generate_data(full_dataset, thread_nb):
    ipt = int(len(full_dataset) / thread_nb)
    dfs = np.array_split(full_dataset, thread_nb)
    pathes = [f"./{l}.csv" for l in range(thread_nb)]
    for l in range(thread_nb):
        dfs[l].to_csv(pathes[l], index=None)
    for i in range(thread_nb):
        x = threading.Thread(target=generate_target, args=(pathes[i], ))
        x.start()


generate_data(df, 5)

# for dset in ["train", "test"]:
#     df = pd.read_csv(f"{data_path}datasets/{dset}.csv", header=[0, 1])
#     for i in tqdm(range(len(df))):
#         image_path = df[("Inputs", "images")][i].replace("/data/", "/data/icar/")
#         name = df[("Outputs", "boxes")][i].split("/")[-1].split(".")[0]
#         if not os.path.exists(image_path):
#             print(name)
#             if "Patrick_t001" in image_path:
#                 df[("Inputs", "images")][i] = df[("Inputs", "images")][i].replace("t001", "t002")
#                 df[("Inputs", "rpn_match")][i] = df[("Inputs", "rpn_match")][i].replace("t001", "t002")
#                 df[("Inputs", "rpn_bbox")][i] = df[("Inputs", "rpn_bbox")][i].replace("t001", "t002")
#                 df[("Outputs", "segs")][i] = df[("Outputs", "segs")][i].replace("t001", "t002")
#                 df[("Outputs", "boxes")][i] = df[("Outputs", "boxes")][i].replace("t001", "t002")
#                 df[("Outputs", "minimasks")][i] = df[("Outputs", "minimasks")][i].replace("t001", "t002")
#                 generate_unique_target(name, df[("Outputs", "boxes")][i].replace("/data/", "/data/icar/").replace("t001", "t002"))
#             elif "Patrick_t050" in image_path:
#                 df[("Inputs", "images")][i] = df[("Inputs", "images")][i].replace("t050", "t051")
#                 df[("Inputs", "rpn_match")][i] = df[("Inputs", "rpn_match")][i].replace("t050", "t051")
#                 df[("Inputs", "rpn_bbox")][i] = df[("Inputs", "rpn_bbox")][i].replace("t050", "t051")
#                 df[("Outputs", "segs")][i] = df[("Outputs", "segs")][i].replace("t050", "t051")
#                 df[("Outputs", "boxes")][i] = df[("Outputs", "boxes")][i].replace("t050", "t051")
#                 df[("Outputs", "minimasks")][i] = df[("Outputs", "minimasks")][i].replace("t050", "t051")
#                 generate_unique_target(name,
#                                        df[("Outputs", "boxes")][i].replace("/data/", "/data/icar/").replace("t050", "t051"))
#             else:
#                 print("FUCK", name)
#     df.to_csv(f"{data_path}datasets/{dset}.csv", index=None)


# for dset in ["train", "test"]:
#     df = pd.read_csv(f"{data_path}datasets/{dset}.csv", header=[0, 1])
#     for i in tqdm(range(len(df))):
#         image_path = df[("Inputs", "images")][i].replace("/data/", "/data/icar/")
#         name = df[("Outputs", "boxes")][i].split("/")[-1].split(".")[0]
#         if "Patrick_t002" in image_path:
#             print(name)
#             generate_unique_target(name, df[("Outputs", "boxes")][i].replace("/data/", "/data/icar/"))
#         elif "Patrick_t051" in image_path:
#             print(name)
#             generate_unique_target(name, df[("Outputs", "boxes")][i].replace("/data/", "/data/icar/"))

# for set in ["train", "test"]:
#     for i in tqdm(range(len(set))):
#         name = set[("Outputs", "boxes")][i].split("/")[-1].split(".")[0]
#         gt_boxes = np.loadtxt(dataset[("Outputs", "boxes")][i])
#         gt_class_ids = np.ones(gt_boxes.shape[0])
#         rpn_match, rpn_bbox = build_rpn_targets((256, 256, 256), anchors, gt_class_ids, gt_boxes, config)
#         match_path = f"{data_path}rpn_match/{name}.npy"
#         bbox_path = f"{data_path}rpn_bbox/{name}.npy"
#         np.save(match_path, rpn_match)
#         np.save(bbox_path, rpn_bbox)
#         rpn_match_pathes.append(match_path)
#         rpn_bbox_pathes.append(bbox_path)
#     dataset[("Inputs", "rpn_match")] = pd.Series(rpn_match_pathes, index=dataset.index)
#     dataset[("Inputs", "rpn_bbox")] = pd.Series(rpn_bbox_pathes, index=dataset.index)
#     dataset.to_csv(f"{config.DATASET_DIR}{set}.csv", index=None)
