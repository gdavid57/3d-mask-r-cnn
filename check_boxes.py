# import os
# import plotly.express as px
# from tqdm import tqdm
import numpy as np
from sklearn.cluster import KMeans
import pandas as pd
import random


### RAW DATASET

# list_box_files = os.listdir("raw_boxes")

# dataset = pd.DataFrame({
#     "name": [],
#     "embryo": [],
#     "instance_nb": [],
#     "typical": [],
# })

# for box_file in tqdm(list_box_files):
#     loaded_boxes = np.loadtxt(f"raw_boxes/{box_file}")
#     if loaded_boxes.shape[0] < 300:
#         dx = loaded_boxes[:, 4] - loaded_boxes[:, 1]
#         dy = loaded_boxes[:, 3] - loaded_boxes[:, 0]
#         dz = loaded_boxes[:, 5] - loaded_boxes[:, 2]
#         typical = np.mean(np.power(dx*dy*dz, 1/3))
#         if typical < 80:
#             xz = np.minimum(dx, dz) / np.maximum(dx, dz)
#             xy = np.minimum(dx, dy) / np.maximum(dx, dy)
#             yz = np.minimum(dy, dz) / np.maximum(dy, dz)

#             distance = np.sqrt((xy - (xy+xz+yz)/3)**2 + (xz - (xy+xz+yz)/3)**2 + (yz - (xy+xz+yz)/3)**2)

#             name = box_file.split("/")[-1].split('_a')[0]
#             embryo = box_file.split("/")[-1].split('_')[0]

#             dataset.loc[len(dataset)] = [name, embryo, loaded_boxes.shape[0], typical]

# dataset.to_csv("global_dataset.csv", index=None)


### DATASETS

dataset = pd.read_csv("global_dataset.csv", header=[0])

kmeans = KMeans(3).fit(np.asarray(dataset["typical"]).reshape(-1, 1))

labels = kmeans.labels_
dataset["label"] = labels

### TRAIN DATASET

train_examples = dataset[dataset["embryo"] != "Patrick"]

labels, counts = np.unique(train_examples["label"], return_counts=True)

minimum = np.min(counts)

dataset0 = train_examples[train_examples["label"] == labels[0]].sample(n=minimum)
dataset1 = train_examples[train_examples["label"] == labels[1]].sample(n=minimum)
dataset2 = train_examples[train_examples["label"] == labels[2]].sample(n=minimum)

final_dataset = pd.concat([dataset0, dataset1, dataset2], axis=0).reset_index()

data_path = "/data/icar/gdavid/Embryo3D/"

boxes_path = f"{data_path}boxes/"
minimasks_path = f"{data_path}minimasks/"
images_path = f"{data_path}images/"
rpn_matches_path = f"{data_path}rpn_match/"
rpn_bboxes_path = f"{data_path}rpn_bbox/"


train_dataset = pd.DataFrame({
    "images": [],
    "boxes": [],
    "minimasks": [],
    "rpn_match": [],
    "rpn_bbox": [],
    "names": [],
    "embryos": [],
    "instance_nbs": [],
    "typicals": [],
    "labels": []
})

combis = []
for i in range(8):
    for j in range(6):
        combis.append(f"{i}{j}")

for i in range(len(final_dataset)):
    base_name = final_dataset["name"][i]

    embryo = final_dataset["embryo"][i]
    instance_nb = final_dataset["instance_nb"][i]
    typical = final_dataset["typical"][i]
    label = final_dataset["label"][i]

    random.shuffle(combis)
    special_combis = combis[:12]
    all_augment = [f"{base_name}_a{combi}" for combi in special_combis]
    
    for augment in all_augment:

        box_path = f"{boxes_path}{augment}.dat"
        image_path = f"{images_path}{augment}.tiff"
        minimask_path = f"{minimasks_path}{augment}.pickle"
        rpn_match_path = f"{rpn_matches_path}{augment}.npy"
        rpn_bbox_path = f"{rpn_bboxes_path}{augment}.npy"

        train_dataset.loc[len(train_dataset)] = [image_path, box_path, minimask_path, rpn_match_path, rpn_bbox_path, augment, embryo, instance_nb, typical, label]
    
train_dataset.to_csv("train.csv", index=None)


### TEST DATASET

test_examples = dataset[dataset["embryo"] == "Patrick"].reset_index()

test_dataset = pd.DataFrame({
    "images": [],
    "boxes": [],
    "minimasks": [],
    "rpn_match": [],
    "rpn_bbox": [],
    "names": [],
    "embryos": [],
    "instance_nbs": [],
    "typicals": [],
    "labels": []
})

for i in range(len(test_examples)):
    base_name = test_examples["name"][i]

    embryo = test_examples["embryo"][i]
    instance_nb = test_examples["instance_nb"][i]
    typical = test_examples["typical"][i]
    label = test_examples["label"][i]

    random.shuffle(combis)
    special_combis = combis[:12]

    augment = f"{base_name}_a00"

    box_path = f"{boxes_path}{augment}.dat"
    image_path = f"{images_path}{augment}.tiff"
    minimask_path = f"{minimasks_path}{augment}.pickle"
    rpn_match_path = f"{rpn_matches_path}{augment}.npy"
    rpn_bbox_path = f"{rpn_bboxes_path}{augment}.npy"

    test_dataset.loc[len(test_dataset)] = [image_path, box_path, minimask_path, rpn_match_path, rpn_bbox_path, augment, embryo, instance_nb, typical, label]
    
test_dataset.to_csv("test.csv", index=None)