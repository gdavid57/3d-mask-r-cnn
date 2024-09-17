import os
from tqdm import tqdm
import numpy as np
import pandas as pd

# import random
# from sklearn.cluster import KMeans
# from sklearn.model_selection import train_test_split


### RAW DATASET
# We first build the 'global dataset' that contains
# all the geometrical informations about the cell instance,
# in particular typical sizes.

list_box_files = os.listdir("data/Boxes")

dataset = pd.DataFrame({
    "name": [],
    "embryo": [],
    "instance_nb": [],
    "typical": [],
})

for box_file in tqdm(list_box_files):
    loaded_boxes = np.loadtxt(f"data/Boxes/{box_file}")
    if "PM1" in box_file:
        if loaded_boxes.shape[0] < 350:
            dx = loaded_boxes[:, 4] - loaded_boxes[:, 1]
            dy = loaded_boxes[:, 3] - loaded_boxes[:, 0]
            dz = loaded_boxes[:, 5] - loaded_boxes[:, 2]
            typical = np.mean(np.power(dx*dy*dz, 1/3))

            name = box_file.split("_boxes")[0]
            embryo = box_file.split('_')[0]

            dataset.loc[len(dataset)] = [name, embryo, loaded_boxes.shape[0], typical]
    
    else:
        if loaded_boxes.shape[0] < 300:
            dx = loaded_boxes[:, 4] - loaded_boxes[:, 1]
            dy = loaded_boxes[:, 3] - loaded_boxes[:, 0]
            dz = loaded_boxes[:, 5] - loaded_boxes[:, 2]
            typical = np.mean(np.power(dx*dy*dz, 1/3))
            if typical < 80:

                name = box_file.split(".")[0]
                embryo = box_file.split('_')[0]

                dataset.loc[len(dataset)] = [name, embryo, loaded_boxes.shape[0], typical]

os.makedirs("data/datasets/", exist_ok=True)
dataset.to_csv("data/datasets/global_dataset.csv", index=None)
print("Global dataset saved.")

### DATASETS
# We then separate the training examples according to three clusters 
# in regards to the average typical cell sizes in an image.

# dataset = pd.read_csv("global_dataset.csv", header=[0])

# kmeans = KMeans(3).fit(np.asarray(dataset["typical"]).reshape(-1, 1))

# labels = kmeans.labels_
# dataset["label"] = labels

### TRAIN AND VALID DATASET
# We sample from these three categories the same amount of 
# example. This constitutes our training and validation pools.

# train_examples = dataset[dataset["embryo"] != "PM1"]

# labels, counts = np.unique(train_examples["label"], return_counts=True)

# minimum = np.min(counts)
# print(f"Smallest category contains {minimum} elements.")

# dataset0 = train_examples[train_examples["label"] == labels[0]].sample(n=minimum)
# dataset1 = train_examples[train_examples["label"] == labels[1]].sample(n=minimum)
# dataset2 = train_examples[train_examples["label"] == labels[2]].sample(n=minimum)

# final_dataset = pd.concat([dataset0, dataset1, dataset2], axis=0).reset_index()
# print("Balanced dataset computed.")

data_path = "data/"

boxes_path = f"{data_path}Boxes/"
minimasks_path = f"{data_path}Minimasks/"
images_path = f"{data_path}Inputs/"
rpn_matches_path = f"{data_path}rpn_match/"
rpn_bboxes_path = f"{data_path}rpn_bbox/"
gt_segs_path = f"{data_path}ASTEC_Ground_truth/"


# train_dataset = pd.DataFrame({
#     "images": [],
#     "boxes": [],
#     "minimasks": [],
#     "rpn_match": [],
#     "rpn_bbox": [],
#     "names": [],
#     "embryos": [],
#     "instance_nbs": [],
#     "typicals": [],
#     "labels": []
# })

# combis = []
# for i in range(8):
#     for j in range(6):
#         combis.append(f"{i}{j}")

# combi_nb = 24
# for i in tqdm(range(len(final_dataset))):
#     base_name = final_dataset["name"][i]

#     embryo = final_dataset["embryo"][i]
#     instance_nb = final_dataset["instance_nb"][i]
#     typical = final_dataset["typical"][i]
#     label = final_dataset["label"][i]

#     random.shuffle(combis)
#     combis = combis[:combi_nb]
#     all_augment = [f"{base_name}_a{combi}" for combi in combis]
    
#     for augment in all_augment:

#         box_path = f"{boxes_path}{augment}.dat"
#         image_path = f"{images_path}{augment}.tiff"
#         minimask_path = f"{minimasks_path}{augment}.pickle"
#         rpn_match_path = f"{rpn_matches_path}{augment}.npy"
#         rpn_bbox_path = f"{rpn_bboxes_path}{augment}.npy"

#         train_dataset.loc[len(train_dataset)] = [image_path, box_path, minimask_path, rpn_match_path, rpn_bbox_path, augment, embryo, instance_nb, typical, label]


# train_dataset, valid_dataset = train_test_split(train_dataset, test_size=0.01, shuffle=True)

# train_dataset.to_csv("train.csv", index=None)
# valid_dataset.to_csv("valid.csv", index=None)
# print("Train and valid datasets saved.")


### TEST DATASET
# The test set is composed of the remarkable embryo
# series called PM1.

test_examples = dataset[dataset["embryo"] == "PM1"].reset_index()

test_dataset = pd.DataFrame({
    "images": [],
    "gt_segs": [],
    "boxes": [],
    "minimasks": [],
    "rpn_match": [],
    "rpn_bbox": [],
    "names": [],
    "embryos": [],
    "instance_nbs": [],
    "typicals": [],
    # "labels": []
})

for i in tqdm(range(len(test_examples))):
    base_name = test_examples["name"][i]

    embryo = test_examples["embryo"][i]
    instance_nb = test_examples["instance_nb"][i]
    typical = test_examples["typical"][i]
    # label = test_examples["label"][i]

    box_path = f"{boxes_path}{base_name}_boxes.dat"
    image_path = f"{images_path}{base_name}_input.tiff"
    minimask_path = f"{minimasks_path}{base_name}_minimasks.pickle"
    rpn_match_path = f"{rpn_matches_path}{base_name}_rpn_match.npy"
    rpn_bbox_path = f"{rpn_bboxes_path}{base_name}_rpn_bbox.npy"
    gt_seg_path = f"{gt_segs_path}{base_name}_ASTEC.tiff"


    # test_dataset.loc[len(test_dataset)] = [image_path, box_path, minimask_path, rpn_match_path, rpn_bbox_path, base_name, embryo, instance_nb, typical, label]
    test_dataset.loc[len(test_dataset)] = [image_path, gt_seg_path, box_path, minimask_path, rpn_match_path, rpn_bbox_path, base_name, embryo, instance_nb, typical]

test_dataset.to_csv("data/datasets/test.csv", index=None)
print("Test dataset saved.")