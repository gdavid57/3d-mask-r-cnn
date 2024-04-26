import os
import random
import numpy as np
from skimage import io
import nibabel as nib
import gzip
import shutil
import bz2
import _pickle as cPickle
from skimage.exposure import is_low_contrast
from skimage.transform import resize
# import threading
import time
import pandas as pd
from tqdm import tqdm
from sklearn.cluster import KMeans


dataset_path = f"/data/gdavid/Embryo3D/"
images_path = f"{dataset_path}images/"
boxes_path = f"{dataset_path}boxes/"
minimasks_path = f"{dataset_path}minimasks/"
segs_path = f"{dataset_path}segs/"

images_list = [f"{images_path}{x}" for x in os.listdir(images_path) if "Eminie" not in x]
cell_nbs = []
for image_path in tqdm(images_list.copy()):
    boxes = np.loadtxt(image_path.replace("images", "boxes").replace(".tiff", ".dat"))
    if boxes.shape[0] > 300:
        images_list.remove(image_path)
    else:
        cell_nbs.append(boxes.shape[0])

labels = KMeans(4).fit(np.asarray(cell_nbs).reshape(-1, 1)).labels_
unique_labels, class_counts = np.unique(labels, return_counts=True)
min_class_size = np.min(class_counts)
balanced_data = []
for label in unique_labels:
    class_indices = np.where(labels == label)[0]
    random_indices = np.random.choice(class_indices, min_class_size, replace=False)
    balanced_data.extend([images_list[i] for i in random_indices])

images_list = balanced_data
matches_list = [x.replace(".tiff", ".npy").replace("images", "rpn_match") for x in images_list]
bbox_list = [x.replace(".tiff", ".npy").replace("images", "rpn_bbox") for x in images_list]
boxes_list = [x.replace(".tiff", ".dat").replace("images", "boxes") for x in images_list]
minimasks_list = [x.replace(".tiff", ".pickle").replace("images", "minimasks") for x in images_list]
segs_list = [x.replace("images", "minimasks") for x in images_list]


n = len(images_list)
rang = list(range(n))
random.shuffle(rang)
rang = rang[:11000]

images_list = [images_list[i] for i in rang]
matches_list = [matches_list[i] for i in rang]
bbox_list = [bbox_list[i] for i in rang]
boxes_list = [boxes_list[i] for i in rang]
minimasks_list = [minimasks_list[i] for i in rang]
segs_list = [segs_list[i] for i in rang]

# n = 20000

train_size = 10000
# valid_size = int(n * 0.05)
test_size = 1000
print(train_size, test_size)
for name, ind_max in zip(["train", "test"], [train_size, test_size]):
    images_serie = pd.Series(images_list[:ind_max], index=range(ind_max))
    matches_serie = pd.Series(matches_list[:ind_max], index=range(ind_max))
    bbox_serie = pd.Series(bbox_list[:ind_max], index=range(ind_max))
    segs_serie = pd.Series(segs_list[:ind_max], index=range(ind_max))
    boxes_serie = pd.Series(boxes_list[:ind_max], index=range(ind_max))
    minimasks_serie = pd.Series(minimasks_list[:ind_max], index=range(ind_max))
    del images_list[:ind_max]
    del matches_list[:ind_max]
    del bbox_list[:ind_max]
    del boxes_list[:ind_max]
    del minimasks_list[:ind_max]
    del segs_list[:ind_max]
    df = {("Inputs", "images"): images_serie, ("Inputs", "rpn_match"): matches_serie,
          ("Inputs", "rpn_bbox"): bbox_serie, ("Outputs", "segs"): segs_serie, ("Outputs", "boxes"): boxes_serie,
          ("Outputs", "minimasks"): minimasks_serie}
    df = pd.DataFrame(df)
    df.to_csv(f"{dataset_path}datasets/{name}.csv", index=None)
