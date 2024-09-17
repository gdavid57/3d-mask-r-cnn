import os
import numpy as np
from skimage import io
import bz2
import _pickle as cPickle
import time
import threading


def compute_cube_summit_transform(matrices):
    results = []
    for matrix in matrices:
        results.append(matrix)
        transform = np.rot90( np.rot90(matrix, axes=(0,2) ), axes=(1,0) )
        results.append(transform)
        transform = np.rot90( np.rot90(transform, axes=(0,2) ), axes=(1,0) )
        results.append(transform)
        transform = np.rot90( np.flip(matrix, axis=0), axes=(2,0))
        results.append(transform)
        transform = np.rot90(np.rot90(transform, axes=(0, 2)), axes=(1, 0))
        results.append(transform)
        transform = np.rot90(np.rot90(transform, axes=(0, 2)), axes=(1, 0))
        results.append(transform)
    return results


def compute_cube_summit(matrix):
    results = [matrix]
    transform = np.rot90(matrix, axes=(2, 0))
    results.append(transform)
    transform = np.rot90(transform, axes=(2, 0))
    results.append(transform)
    transform = np.rot90(transform, axes=(2, 0))
    results.append(transform)
    transform = np.rot90(matrix, k=2, axes=(0,1))
    results.append(transform)
    transform = np.rot90(transform, axes=(2, 0))
    results.append(transform)
    transform = np.rot90(transform, axes=(2, 0))
    results.append(transform)
    transform = np.rot90(transform, axes=(2, 0))
    results.append(transform)
    return results


def identity(x, y):
    return x


def compute_transform(image, list_func_arg):
    result = np.copy(image)
    for [func, arg] in list_func_arg:
        if type(arg) != int:
            result = func(result, *arg)
        else:
            result = func(result, arg)
    return result


def make_augmentation(image, augmentation, list_summit, list_transform):
    summit = int(augmentation[-2])
    transform = int(augmentation[-1])
    new_image = compute_transform(image, list_summit[summit])
    new_image = compute_transform(new_image, list_transform[transform])
    return new_image


def compute_boxes_transform(ori_boxes, indices):
    boxes = np.zeros((ori_boxes.shape[0], 6))
    mask_indices = np.hstack((np.asarray(indices[:3]) >= 3, np.asarray(indices[3:]) <= 2))
    for index in range(6):
        if mask_indices[index]:
            boxes[:, index] = 256 - ori_boxes[:, indices[index]]
        else:
            boxes[:, index] = ori_boxes[:, indices[index]]
    return boxes


def make_boxes_augmentation(ori_boxes, augmentation, list_boxes_summit, list_boxes_transform):
    boxes_summit = int(augmentation[-2])
    boxes_transform = int(augmentation[-1])
    new_boxes = compute_boxes_transform(ori_boxes, list_boxes_summit[boxes_summit])
    new_boxes = compute_boxes_transform(new_boxes, list_boxes_transform[boxes_transform])
    return new_boxes


def groundtruth_generator(masks, boxes, augmentation, list_summit, list_transform, list_boxes_summit,
                              list_boxes_transform):
    num_cells = masks.shape[-1]
    for c in range(num_cells):
        masks[..., c] = make_augmentation(masks[..., c], augmentation, list_summit, list_transform)
    boxes = make_boxes_augmentation(boxes, augmentation, list_boxes_summit, list_boxes_transform)
    return masks, boxes


def augmentator(datadir, outputpath, list_augmentations, list_summit, list_transform, list_boxes_summit,
                    list_boxes_transform):
    start = time.time()
    while list_augmentations:
        augmentation = list_augmentations[0]
        list_augmentations.remove(augmentation)
        print(augmentation + ' in progress...')
        image = io.imread(datadir + 'Inputs/' + augmentation[:-4] + '.tiff')
        seg = io.imread(datadir + 'ASTEC_Ground_truth/' + augmentation[:-4] + '.tiff')
        image = make_augmentation(image, augmentation, list_summit, list_transform)
        seg = make_augmentation(seg, augmentation, list_summit, list_transform)
        masks = bz2.BZ2File(datadir + 'Minimasks/' + augmentation[:-4] + '.pickle', 'rb')
        masks = cPickle.load(masks)
        boxes = np.loadtxt(datadir + 'Boxes/' + augmentation[:-4] + '.dat')
        masks, boxes = groundtruth_generator(masks, boxes, augmentation, list_summit, list_transform,
                                                 list_boxes_summit, list_boxes_transform)
        print(augmentation + ' is augmented.')
        io.imsave(outputpath + 'Inputs/' + augmentation + '.tiff', image)
        io.imsave(outputpath + 'ASTEC_Ground_truth/' + augmentation + '.tiff', seg)
        with bz2.BZ2File(outputpath + 'Minimasks/' + augmentation + '.pickle', 'w') as f:
            cPickle.dump(masks, f)
        np.savetxt(outputpath + 'Boxes/' + augmentation + '.dat', boxes)
        print(augmentation + ' is done.')
    end = time.time()
    print(end-start)


def compute_mask_transform(ori_masks, indices):
    new_masks = []
    mask_indices = (np.asarray(indices[:3]) >= 3)
    for arg_mask in ori_masks:
        new_arg_mask = np.zeros(arg_mask.shape).astype(np.uint8)
        for index in range(3):
            if mask_indices[index]:
                new_arg_mask[:, index] = 256 - arg_mask[:, indices[index] % 3 ]
            else:
                new_arg_mask[:, index] = arg_mask[:, indices[index] % 3 ]
        new_masks.append(new_arg_mask)
    return new_masks


def make_mask_augmentation(ori_masks, augmentation, list_boxes_summit, list_boxes_transform):
    boxes_summit = int(augmentation[-2])
    boxes_transform = int(augmentation[-1])
    new_masks = compute_mask_transform(ori_masks, list_boxes_summit[boxes_summit][:3])
    new_masks = compute_mask_transform(new_masks, list_boxes_transform[boxes_transform][:3])
    return new_masks


def mask_augmentator(datadir, outputpath, list_augmentations, list_boxes_summit, list_boxes_transform):
    start = time.time()
    while list_augmentations:
        augmentation = list_augmentations[0]
        list_augmentations.remove(augmentation)
        print(augmentation + ' in progress...')
        masks = bz2.BZ2File(datadir + augmentation[:-4] + '.pickle', 'rb')
        masks = cPickle.load(masks)
        new_masks = make_mask_augmentation(masks, augmentation, list_boxes_summit, list_boxes_transform)
        print(augmentation + ' is augmented.')
        with bz2.BZ2File(outputpath + augmentation + '.pickle', 'w') as f:
            cPickle.dump(new_masks, f)
        print(augmentation + ' is done.')
    end = time.time()
    print(end-start)


list_summit = [ [[identity, 0]],
                [[np.rot90, (1, (0, 2))]],
                [[np.rot90, (2, (0, 2))]],
                [[np.rot90, (1, (2, 0))]],
                [[np.rot90, (2, (0, 1))]],
                [[np.rot90, (2, (0, 1))], [np.rot90, (1, (0, 2))]],
                [[np.rot90, (2, (0, 1))], [np.rot90, (2, (0, 2))]],
                [[np.rot90, (2, (0, 1))], [np.rot90, (1, (2, 0))]] ]
list_transform = [ [[identity, 0]],
                   [[np.rot90, (1, (0, 2))], [np.rot90, (1, (0, 1))]],
                   [[np.rot90, (1, (0, 2))], [np.rot90, (1, (0, 1))], [np.rot90, (1, (0, 2))], [np.rot90, (1, (0, 1))]],
                   [[np.flip, 0], [np.rot90, (1, (2, 0))]],
                   [[np.flip, 0], [np.rot90, (1, (2, 0))], [np.rot90, (1, (0, 2))], [np.rot90, (1, (0, 1))]],
                   [[np.flip, 0], [np.rot90, (1, (2, 0))], [np.rot90, (1, (0, 2))], [np.rot90, (1, (0, 1))],
                    [np.rot90, (1, (0, 2))], [np.rot90, (1, (0, 1))]] ]
list_boxes_summit = [ [0, 1, 2, 3, 4, 5], [5, 1, 0, 2, 4, 3], [3, 1, 5, 0, 4, 2], [2, 1, 3, 5, 4, 0],
                      [3, 4, 2, 0, 1, 5], [5, 4, 3, 2, 1, 0], [0, 4, 5, 3, 1, 2], [2, 4, 0, 5, 1, 3] ]
list_boxes_transform = [ [0, 1, 2, 3, 4, 5], [4, 5, 0, 1, 2, 3], [2, 3, 4, 5, 0, 1],
                         [2, 1, 0, 5, 4, 3], [4, 3, 2, 1, 0, 5], [0, 5, 4, 3, 2, 1] ]

datadir = "data/"
outputpath = "data/"

list_images = os.listdir(datadir + "Inputs/")
list_augmentations = []

for image in list_images:
    for i in range(8):
        for j in range(6):
            list_augmentations.append(image[:-5]+'_a'+str(i)+str(j))
print(len(list_augmentations))

nb_thread = 5
for i in range(nb_thread):
    x = threading.Thread(target=augmentator, args=(datadir, outputpath, list_augmentations, list_summit, list_transform, list_boxes_summit,
                    list_boxes_transform))
    x.start()
