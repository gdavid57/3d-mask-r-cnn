import numpy as np
import pandas as pd
from tqdm import tqdm
import os
from sklearn.cluster import KMeans


def compute_max_and_mean_typical_sizes(boxes):
    volumes = (boxes[:, 3] - boxes[:, 0]) * (boxes[:, 4] - boxes[:, 1]) * (boxes[:, 5] - boxes[:, 2])
    typical_sizes = np.power(volumes, 1/3)
    # print(typical_sizes.shape)
    return np.max(typical_sizes), np.mean(typical_sizes)


def compute_aspect_ratio(boxes):
    dx = boxes[:, 4] - boxes[:, 1]
    dy = boxes[:, 3] - boxes[:, 0]
    dz = boxes[:, 5] - boxes[:, 2]
    xz = np.minimum(dx, dz) / np.maximum(dx, dz)
    xy = np.minimum(dx, dy) / np.maximum(dx, dy)
    yz = np.minimum(dy, dz) / np.maximum(dy, dz)
    return np.mean(xy), np.mean(xz), np.mean(yz)

def make_dataframe(dataset_path, test_dataframe, max_instances, label_nb, criterion):
    images_path = f"{dataset_path}images/"
    boxes_path = f"{dataset_path}boxes/"
    minimasks_path = f"{dataset_path}minimasks/"
    segs_path = f"{dataset_path}segs/"
    rpnmatch_path = f"{dataset_path}rpn_match_bis/"
    rpnbbox_path = f"{dataset_path}rpn_bbox_bis/"

    list_boxes = [x for x in os.listdir(boxes_path) if "Eminie" not in x]
    list_test_boxes = [x.split("/")[-1] for x in list(test_dataframe[("Outputs", "boxes")])]
    for x in [f"Patrick_t{str(y).zfill(3)}_a00.dat" for y in range(1, 193)]:
        list_boxes.remove(x)
    list_boxes = [x for x in list_boxes if x not in list_test_boxes]
    instance_nb = []
    maxes = []
    means = []
    xys, xzs, yzs = [], [], []
    names = []
    for box_path in tqdm(list_boxes):
        boxes = np.loadtxt(f"{boxes_path}{box_path}")
        if boxes.shape[0] < max_instances:
            instance_nb.append(boxes.shape[0])
            max, mean = compute_max_and_mean_typical_sizes(boxes)
            maxes.append(max)
            means.append(mean)
            xy, xz, yz = compute_aspect_ratio(boxes)
            xys.append(xy)
            xzs.append(xz)
            yzs.append(yz)
            names.append(box_path.split("/")[-1].split(".")[0])

    kmeans = KMeans(label_nb).fit(np.asarray(instance_nb).reshape(-1, 1))
    labels = kmeans.labels_
    clusters = kmeans.cluster_centers_
    argsort = np.argsort(np.argsort(clusters.flatten()))
    print(clusters, argsort)

    df = {
        ("Inputs", "images"): pd.Series([f"{images_path}{name}.tiff" for name in names]),
        ("Inputs", "rpn_match"): pd.Series([f"{rpnmatch_path}{name}.npy" for name in names]),
        ("Inputs", "rpn_bbox"): pd.Series([f"{rpnbbox_path}{name}.npy" for name in names]),
        ("Outputs", "segs"): pd.Series([f"{segs_path}{name}.tiff" for name in names]),
        ("Outputs", "boxes"): pd.Series([f"{boxes_path}{name}.dat" for name in names]),
        ("Outputs", "minimasks"): pd.Series([f"{minimasks_path}{name}.pickle" for name in names]),
        ("Derivatives", "names"): pd.Series(names),
        ("Derivatives", "instanb"): pd.Series(instance_nb),
        ("Derivatives", "labels"): pd.Series(labels),
        ("Derivatives", "maxes"): pd.Series(maxes),
        ("Derivatives", "means"): pd.Series(means),
        ("Derivatives", "xys"): pd.Series(xys),
        ("Derivatives", "xzs"): pd.Series(xzs),
        ("Derivatives", "yzs"): pd.Series(yzs)
    }
    df = pd.DataFrame(df)
    df[("Derivatives", "distance")] = np.sqrt(
        (df[("Derivatives", "xys")] - (df[("Derivatives", "xys")] + df[("Derivatives", "xzs")] + df[("Derivatives", "yzs")]) / 3)**2 +
        (df[("Derivatives", "xzs")] - (df[("Derivatives", "xys")] + df[("Derivatives", "xzs")] + df[("Derivatives", "yzs")]) / 3)**2 +
        (df[("Derivatives", "yzs")] - (df[("Derivatives", "xys")] + df[("Derivatives", "xzs")] + df[("Derivatives", "yzs")]) / 3)**2
    )
    df[("Derivatives", "criterion")] = df[("Derivatives", "distance")] < criterion

    return df, argsort


def complete_dataframe(dataframe, dataset_path, max_instances, label_nb, criterion):
    boxes_path = f"{dataset_path}boxes/"

    list_boxes = [x.split("/")[-1] for x in dataframe[("Outputs", "boxes")]]
    instance_nb = []
    maxes = []
    means = []
    xys, xzs, yzs = [], [], []
    names = []
    for box_name in tqdm(list_boxes):
        boxes = np.loadtxt(f"{boxes_path}{box_name}")
        if boxes.shape[0] <= max_instances:
            instance_nb.append(boxes.shape[0])
            max, mean = compute_max_and_mean_typical_sizes(boxes)
            maxes.append(max)
            means.append(mean)
            xy, xz, yz = compute_aspect_ratio(boxes)
            xys.append(xy)
            xzs.append(xz)
            yzs.append(yz)
            names.append(box_name.split(".")[0])
    kmeans = KMeans(label_nb).fit(np.asarray(instance_nb).reshape(-1, 1))
    labels = kmeans.labels_
    print(kmeans.cluster_centers_)

    df = dataframe.copy()
    # print(len(names), len(xys), len(df))
    # df[("Derivatives", "names")] = pd.Series(names),
    df[("Derivatives", "instanb")] = pd.Series(instance_nb)
    df[("Derivatives", "labels")] = pd.Series(labels)
    df[("Derivatives", "maxes")] = pd.Series(maxes)
    df[("Derivatives", "means")] = pd.Series(means)
    df[("Derivatives", "xys")] = pd.Series(xys)
    df[("Derivatives", "xzs")] = pd.Series(xzs)
    df[("Derivatives", "yzs")] = pd.Series(yzs)

    df[("Derivatives", "distance")] = np.sqrt(
        (df[("Derivatives", "xys")] - (
                    df[("Derivatives", "xys")] + df[("Derivatives", "xzs")] + df[("Derivatives", "yzs")]) / 3) ** 2 +
        (df[("Derivatives", "xzs")] - (
                    df[("Derivatives", "xys")] + df[("Derivatives", "xzs")] + df[("Derivatives", "yzs")]) / 3) ** 2 +
        (df[("Derivatives", "yzs")] - (
                    df[("Derivatives", "xys")] + df[("Derivatives", "xzs")] + df[("Derivatives", "yzs")]) / 3) ** 2
    )
    df[("Derivatives", "criterion")] = df[("Derivatives", "distance")] < criterion

    return df


def pick_dataframe(dataframe, argsort, rules):
    samples = []
    for criterion in dataframe[("Derivatives", "criterion")].unique():
        df = dataframe[dataframe[("Derivatives", "criterion")] == criterion]
        for label in dataframe[("Derivatives", "labels")].unique():
            dff = df[df[("Derivatives", "labels")] == label]
            samples.append(dff.sample(n=rules[(criterion, argsort[label])]))
    balanced = pd.concat(samples, axis=0)
    return balanced.reset_index()


#   Local
# dataset_path = f"/home/gdavid/Projets/DATA/"
# test_set = pd.read_csv(f"./test.csv", header=[0, 1])

#   AC922
dataset_path = "/data/icar/gdavid/Embryo3D/"
test_set = pd.read_csv(f"{dataset_path}datasets/test.csv", header=[0, 1])


dataframe, argsort = make_dataframe(dataset_path, test_set,300, 4, 0.0365)
dataframe.to_csv(f"{dataset_path}datasets/train_head_full.csv", index=None)
# print(len(dataframe))

for criterion in dataframe[("Derivatives", "criterion")].unique():
    for label in dataframe[("Derivatives", "labels")].unique():
        df = dataframe[dataframe[("Derivatives", "labels")] == label]
        dff = df[df[("Derivatives", "criterion")] == criterion]
        print(label, criterion, len(dff))


# for criterion in dataframe[("Derivatives", "criterion")].unique():
#     df = dataframe[dataframe[("Derivatives", "criterion")] == criterion]
#     print(criterion, len(df))
# for label in dataframe[("Derivatives", "labels")].unique():
#     df = dataframe[dataframe[("Derivatives", "labels")] == label]
#     print(label, len(df))


# train_set = pd.read_csv("./train.csv", header=[0, 1])
# test_set = pd.read_csv("./test.csv", header=[0, 1])

# datasets = pd.concat([train_set, test_set], axis=0).reset_index()
# dataframe = complete_dataframe(datasets, dataset_path, 300, 4, 0.0365)
# print(len(dataframe))
#
# for criterion in dataframe[("Derivatives", "criterion")].unique():
#     for label in dataframe[("Derivatives", "labels")].unique():
#         df = dataframe[dataframe[("Derivatives", "labels")] == label]
#         dff = df[df[("Derivatives", "criterion")] == criterion]
#         print(label, criterion, len(dff))

# for criterion in dataframe[("Derivatives", "criterion")].unique():
#     df = dataframe[dataframe[("Derivatives", "criterion")] == criterion]
#     print(criterion, len(df))
# for label in dataframe[("Derivatives", "labels")].unique():
#     df = dataframe[dataframe[("Derivatives", "labels")] == label]
#     print(label, len(df))


# rules = {(True, 0): 48, (True, 1): 48, (True, 2): 2452, (True, 3): 2455,
#          (False, 0): 2452, (False, 1): 2452, (False, 2): 48, (False, 3): 45}

# rules = {(True, 0): 45, (True, 1): 48, (True, 2): 13927, (True, 3): 5991,
#          (False, 0): 13424, (False, 1): 6464, (False, 2): 48, (False, 3): 45}

# picked = pick_dataframe(dataframe, argsort, rules)
# # for criterion in picked[("Derivatives", "criterion")].unique():
# #     for label in picked[("Derivatives", "labels")].unique():
# #         df = picked[picked[("Derivatives", "labels")] == label]
# #         dff = df[df[("Derivatives", "criterion")] == criterion]
# #         print(label, criterion, len(dff))

# picked = picked.sample(frac=1.0, random_state=1)
# picked.to_csv(f"{dataset_path}datasets/train_head.csv", index=None)
