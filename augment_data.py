import os

import matplotlib.pyplot as plt
import numpy as np
from skimage import io
import bz2
import _pickle as cPickle
from scipy.ndimage import rotate
from skimage.transform import resize
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
        pathes = []
        augmentation = list_augmentations[0]
        list_augmentations.remove(augmentation)
        print(augmentation + ' in progress...')
        image = io.imread(datadir + 'images/' + augmentation[:-4] + '.tiff')
        seg = io.imread(datadir + 'segs/' + augmentation[:-4] + '.tiff')
        image = make_augmentation(image, augmentation, list_summit, list_transform)
        seg = make_augmentation(seg, augmentation, list_summit, list_transform)
        masks = bz2.BZ2File(datadir + 'minimasks/' + augmentation[:-4] + '.pickle', 'rb')
        masks = cPickle.load(masks)
        boxes = np.loadtxt(datadir + 'boxes/' + augmentation[:-4] + '.dat')
        masks, boxes = groundtruth_generator(masks, boxes, augmentation, list_summit, list_transform,
                                                 list_boxes_summit, list_boxes_transform)
        print(augmentation + ' is augmented.')
        io.imsave(outputpath + 'images/' + augmentation + '.tiff', image)
        # pathes.append(outputpath + 'images/' + augmentation + '.tiff')
        io.imsave(outputpath + 'segs/' + augmentation + '.tiff', seg)
        # pathes.append(outputpath + 'segs/' + augmentation + '.tiff')
        with bz2.BZ2File(outputpath + 'minimasks/' + augmentation + '.pickle', 'w') as f:
            cPickle.dump(masks, f)
        # pathes.append(outputpath + 'minimasks/' + augmentation + '.pickle')
        np.savetxt(outputpath + 'boxes/' + augmentation + '.dat', boxes)
        # pathes.append(outputpath + 'boxes/' + augmentation + '.dat')
        print(augmentation + ' is done.')
        # os.system('scp ' + outputpath + 'images/' + augmentation +
        #           '.tiff ukw72um@jean-zay.idris.fr:/gpfsscratch/rech/umm/ukw72um/data/Embryo3D/images/')
        # os.system('scp ' + outputpath + 'segs/' + augmentation +
        #           '.tiff ukw72um@jean-zay.idris.fr:/gpfsscratch/rech/umm/ukw72um/data/Embryo3D/segs/')
        # os.system('scp ' + outputpath + 'minimasks/' + augmentation +
        #           '.pickle ukw72um@jean-zay.idris.fr:/gpfsscratch/rech/umm/ukw72um/data/Embryo3D/minimasks/')
        # os.system('scp ' + outputpath + 'classes_and_boxes/' + augmentation +
        #           '.dat ukw72um@jean-zay.idris.fr:/gpfsscratch/rech/umm/ukw72um/data/Embryo3D/classes_and_boxes/')
        # os.remove(outputpath + 'images/' + augmentation + '.tiff')
        # os.remove(outputpath + 'segs/' + augmentation + '.tiff')
        # os.remove(outputpath + 'minimasks/' + augmentation + '.pickle')
        # os.remove(outputpath + 'classes_and_boxes/' + augmentation + '.dat')
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

# datadir = './data/'
# datadir = './masks/'
datadir = "/home/gdavid/Projets/DATA/Patrick_a00/"
# outputpath = './data/augmented/'
# outputpath = '/data/icar/gdavid/Embryo3D/masks/'
outputpath = "/home/gdavid/Projets/DATA/Patrick/"

list_images = os.listdir(datadir + "images/")
list_augmentations = []
# with open('bonjour.dat', 'r') as f:
#     already_done = f.readlines()
for image in list_images:
    for i in range(8):
        for j in range(6):
            list_augmentations.append(image[:-5]+'_a'+str(i)+str(j))
print(len(list_augmentations))
# list_augmentations = list_augmentations[:48]

nb_thread = 5
for i in range(nb_thread):
    x = threading.Thread(target=augmentator, args=(datadir, outputpath, list_augmentations, list_summit, list_transform, list_boxes_summit,
                    list_boxes_transform))
    x.start()

# Check data
# seg = io.imread('./data/segs/Patr_t001.tiff')
# new_seg = make_augmentation(np.copy(seg), 'Patr_t001_a23', list_summit, list_transform)
# print(np.sum(np.where(seg==2,1,0)), np.sum(np.where(new_seg==2,1,0)))
# masks = bz2.BZ2File('./data/minimasks/Patr_t001.pickle', 'rb')
# masks = cPickle.load(masks)
#
# start = time.time()
# new_masks1, boxes1 = groundtruth_generator(new_seg)
# end = time.time()
# print(end-start)
# start = time.time()
# new_masks2, boxes2 = groundtruth_generator_bis(np.copy(masks), 'Patr_t001_a23', list_summit, list_transform)
# end = time.time()
# print(end-start)
#
# for i in range(new_masks1.shape[-1]):
#     if np.sum(new_masks1[..., i]) != np.sum(new_masks2[..., i]) :
#         print(i, abs(np.sum(new_masks1[..., i])-np.sum(new_masks2[..., i])))

# for i in range(boxes1.shape[0]):
#     if (boxes1[i] != boxes2[i]).any() :
#         print(i, boxes1[i], boxes2[i])

# diff = new_masks1[..., 40].astype(np.uint8) - new_masks2[..., 40].astype(np.uint8)
# io.imsave('./diff.tiff', diff)
# io.imsave('./n1.tiff', new_masks1[..., 40].astype(np.uint8))
# io.imsave('./n2.tiff', new_masks2[..., 40].astype(np.uint8))
# io.imsave('./n0.tiff', masks[..., 40].astype(np.uint8))

# start = time.time()
# seg = io.imread('./data/segs/Patr_t001.tiff')
# list_of_labels = np.unique(seg)[1:]
# boxes1 = np.zeros((48 * len(list_of_labels), 6))
# cp=0
# for label in list_of_labels:
#     print(label)
#     for summit in list_summit:
#         for transform in list_transform:
#             new_seg = compute_transform(np.copy(seg), summit)
#             new_seg = compute_transform(new_seg, transform)
#             mask = np.where(new_seg == label, 1, 0).astype(np.uint8)
#             horizontal_indicies = np.where(np.any(np.any(mask, axis=0), axis=1))[0]
#             vertical_indicies = np.where(np.any(np.any(mask, axis=1), axis=1))[0]
#             profound_indicies = np.where(np.any(np.any(mask, axis=0), axis=0))[0]
#             x1, x2 = horizontal_indicies[[0, -1]]
#             y1, y2 = vertical_indicies[[0, -1]]
#             z1, z2 = profound_indicies[[0, -1]]
#             boxes1[cp] = np.array([y1 - 1, x1 - 1, z1 - 1, y2 + 2, x2 + 2, z2 + 2])
#             cp+=1
# end = time.time()
# print(end-start)
#
# start = time.time()
# boxes = np.loadtxt('./data/classes_and_boxes/Patr_t001.dat')
# boxes2 = np.zeros((48 * boxes.shape[0], 6))
# cp=0
# for box in boxes:
#     for boxes_summit in list_boxes_summit:
#         for boxes_transform in list_boxes_transform:
#             new_boxes = compute_boxes_transform(np.copy(box)[np.newaxis,...], boxes_summit)
#             new_boxes = compute_boxes_transform(new_boxes, boxes_transform)
#             boxes2[cp] = new_boxes
#             cp+=1
# end = time.time()
# print(end-start)
#
# for i in range(48 * boxes.shape[0]):
#     if (boxes1[i] != boxes2[i]).any():
#         print(boxes1[i], boxes2[i])

# boxes_dir1 = './data/augmented/classes_and_boxes/'
# boxes_dir2 = './data/augmented2/classes_and_boxes/'
#
# list_boxes1 = os.listdir(boxes_dir1)
# list_boxes1.sort()
# list_boxes2 = os.listdir(boxes_dir2)
# list_boxes2.sort()
#
# for j in range(len(list_boxes1)):
#     boxes1 = np.loadtxt(boxes_dir1 + list_boxes1[j])
#     boxes2 = np.loadtxt(boxes_dir2 + list_boxes2[j])
#     for i in range(boxes1.shape[0]):
#         if (boxes1[i] != boxes2[i]).any():
#             print(list_boxes1[j], list_boxes2[j], i)
#
#
# masks_dir1 = './data/augmented/masks/'
# masks_dir2 = './data/augmented2/masks/'
#
# list_masks1 = os.listdir(masks_dir1)
# list_masks1.sort()
# list_masks2 = os.listdir(masks_dir2)
# list_masks2.sort()
#
# for j in range(len(list_masks1)):
#     masks1 = bz2.BZ2File(masks_dir1 + list_masks1[j], 'rb')
#     masks1 = cPickle.load(masks1)
#     masks2 = bz2.BZ2File(masks_dir2 + list_masks2[j], 'rb')
#     masks2 = cPickle.load(masks2)
#     for i in range(masks1.shape[-1]):
#         if (masks1[..., i] != masks2[..., i]).any():
#             if abs(np.sum(masks1[..., i]) - np.sum(masks2[..., i])) / (56 * 56 * 56) > 0.01:
#                 print(np.sum(masks1[..., i]), np.sum(masks2[..., i]))




# arg_masks = bz2.BZ2File('./masks/Adriana_t001.pickle', 'rb')
# arg_masks = cPickle.load(arg_masks)
# masks = np.zeros((256,256,256,len(arg_masks))).astype(bool)
# for c, arg_mask in enumerate(arg_masks):
#     masks[arg_mask[:, 0], arg_mask[:, 1], arg_mask[:, 2], c] = 1

# io.imsave('./1.tiff', masks[..., 0])
# num_cells = masks.shape[-1]
# for c in range(num_cells):
#     masks[..., c] = make_augmentation(masks[..., c], 'Adriana_t001_a01', list_summit, list_transform)
# io.imsave('./2.tiff', masks[..., 0])
#
# new_arg_bis = np.zeros((arg_masks[0].shape)).astype(np.uint8)
# new_arg_bis[:, 0] = 255 - arg_masks[0][:, 1]
# new_arg_bis[:, 1] = 255 - arg_masks[0][:, 2]
# new_arg_bis[:, 2] = arg_masks[0][:, 0]
#
# new_mask = np.zeros((256,256,256)) .astype(bool)
# new_mask[new_arg_bis[:, 0], new_arg_bis[:, 1], new_arg_bis[:, 2]] = 1
# io.imsave('./3.tiff', new_mask)
# print((new_mask == masks[..., 0]).all())

# list_augmentations = os.listdir('/data/icar/gdavid/Embryo3D/masks/')
# for augmentation in list_augmentations:
#     print(augmentation)
#     augmented_masks = bz2.BZ2File('/data/icar/gdavid/Embryo3D/masks/{}'.format(augmentation), 'rb')
#     augmented_masks = cPickle.load(augmented_masks)
#     new_masks = np.zeros((256, 256, 256, len(augmented_masks))).astype(bool)
#     for k, new_arg_mask in enumerate(augmented_masks):
#         new_masks[augmented_masks[k][:, 0], augmented_masks[k][:, 1], augmented_masks[k][:, 2], k] = 1
#
#     num_cells = masks.shape[-1]
#     new_masks_bis = np.zeros((256, 256, 256, num_cells)).astype(bool)
#     for c in range(num_cells):
#         new_masks_bis[..., c] = make_augmentation(masks[..., c], augmentation[:-7], list_summit, list_transform)
#     print(np.sum(new_masks), np.sum(new_masks_bis), (new_masks_bis == new_masks).all())


# list_augmentations = os.listdir('/data/icar/gdavid/Embryo3D/masks/')
# for augmentation in list_augmentations:
#     print(augmentation)
#     augmented_masks = bz2.BZ2File('/data/icar/gdavid/Embryo3D/masks/{}'.format(augmentation), 'rb')
#     augmented_masks = cPickle.load(augmented_masks)
#     boxes = np.loadtxt('/data/icar/gdavid/Embryo3D/boxes/{}'.format(augmentation.replace('.pickle', '.dat')))
#     for k, new_arg_mask in enumerate(augmented_masks):
#         new_mask = np.zeros((256, 256, 256)).astype(bool)
#         new_mask[augmented_masks[k][:, 0], augmented_masks[k][:, 1], augmented_masks[k][:, 2]] = 1
#         horizontal_indicies = np.where(np.any(np.any(new_mask, axis=0), axis=1))[0]
#         vertical_indicies = np.where(np.any(np.any(new_mask, axis=1), axis=1))[0]
#         profound_indicies = np.where(np.any(np.any(new_mask, axis=0), axis=0))[0]
#         x1, x2 = horizontal_indicies[[0, -1]]
#         y1, y2 = vertical_indicies[[0, -1]]
#         z1, z2 = profound_indicies[[0, -1]]
#         new_boxes = np.array([y1 - 2, x1 - 2, z1 - 2, y2 + 2, x2 + 2, z2 + 2])
#         print(k, (new_boxes == boxes[k]).all())

# for i in range(8):
#     for j in range(6):
#         print(i,j)
#         new_arg_masks = make_mask_augmentation(arg_masks, 'Adriana_t001_a{}{}'.format(i,j), list_boxes_summit, list_boxes_transform)
#         new_masks = np.zeros((256, 256, 256, len(new_arg_masks))).astype(bool)
#         for k, new_arg_mask in enumerate(new_arg_masks):
#             new_masks[new_arg_masks[k][:, 0], new_arg_masks[k][:, 1], new_arg_masks[k][:, 2], k] = 1
#
#         num_cells = masks.shape[-1]
#         new_masks_bis = np.zeros((256, 256, 256, num_cells)).astype(bool)
#         for c in range(num_cells):
#             new_masks_bis[..., c] = make_augmentation(masks[..., c], 'Adriana_t001_a{}{}'.format(i,j), list_summit, list_transform)
#
#         print(len(arg_masks), len(new_arg_masks), arg_masks[0].shape, new_arg_masks[0].shape)
#         print(np.sum(new_masks), np.sum(new_masks_bis), (new_masks_bis == new_masks).all())
