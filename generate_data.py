import os
import numpy as np
from skimage import io
import gzip
import shutil
import bz2
import _pickle as cPickle
from skimage.exposure import is_low_contrast
from skimage.transform import resize
import threading
import time


def get_equal_crop_3D(img, cropped, background=None):
    '''
    Return the equal dimensionalty croped (clear ?)
    '''
    if background is None:
        background = img.min()  # TO FILL WITH BLACK
    [xmin,xmax,ymin,ymax,zmin,zmax]=cropped
    shape=[xmax-xmin,ymax-ymin,zmax-zmin]
    maxS=max(shape)
    if shape[0]<maxS: #X is smaller
        xmin-=round((maxS-shape[0])/2)
        if xmin < 0:
            xmin = 0
        xmax=xmin+maxS
        if maxS<img.shape[0]:  #Crop smaller than X size
            if xmax>img.shape[0]: #We double check that we are inside the coordinates
                xmax = img.shape[0]
                xmin=xmax-maxS
        else:  #Crop Larger than X size
            imfull = np.zeros([maxS, img.shape[1],img.shape[2]], dtype=img.dtype)
            imfull[0:img.shape[0], :, : ] = img
            if background>0:
                imfull[img.shape[0]:maxS, : , :] =background  # FILL WITH BACKGROUND VALUE
            img = imfull
            xmin = 0
            xmax = maxS

    if shape[1]<maxS: #Y is smaller
        ymin -= round((maxS - shape[1]) / 2)
        if ymin < 0:
            ymin = 0
        ymax = ymin + maxS
        if maxS < img.shape[1]:  # Crop smaller than Y size
            if ymax > img.shape[1]:  # We double check that we are inside the coordinates
                ymax = img.shape[1]
                ymin = ymax - maxS
        else:  # Crop Larger than Y size
            imfull = np.zeros([img.shape[0],maxS,img.shape[2]], dtype=img.dtype)
            imfull[:,0:img.shape[1],:] = img
            if background > 0:
                imfull[:,img.shape[1]:maxS, :] =background  # FILL WITH BACKGROUND VALUE
            img = imfull
            ymin = 0
            ymax = maxS

    if shape[1]<maxS: #Z is smaller
        zmin -= round((maxS - shape[2]) / 2)
        if zmin < 0:
            zmin = 0
        zmax = zmin + maxS
        if maxS < img.shape[2]:  # Crop smaller than Z size
            if zmax > img.shape[2]:  # We double check that we are inside the coordinates
                zmax = img.shape[2]
                zmin = zmax - maxS
        else:  # Crop Larger than Z size
            imfull = np.zeros([img.shape[0], img.shape[1],maxS], dtype=img.dtype)
            imfull[:, :,0:img.shape[2]] = img
            if background > 0:
                imfull[:, :, img.shape[2]:maxS] = background  # FILL WITH BACKGROUND VALUE
            img = imfull
            zmin = 0
            zmax = maxS

    return img[xmin:xmax,ymin:ymax,zmin:zmax]


def get_crop_3D(img):
    '''
    RETURN CROP 3D of RAWDATA IMAGE (based on low constrast)
    '''

    #XXXXXXXXX
    xmin=0
    while xmin<img.shape[0] and is_low_contrast(img[xmin,:,:]) :
        xmin+=1
    if xmin>=img.shape[0]-1:
        return [-1,-1,-1,-1,-1,-1]
    xmin -= 1
    if xmin<0:
        xmin=0

    xmax = img.shape[0]-1
    while  xmax>0 and is_low_contrast(img[xmax,:,: ]):
        xmax -= 1
    if xmax<=1:
        return [-1,-1,-1,-1,-1,-1]
    xmax+=1
    if xmax>img.shape[0]-1:
        xmax = img.shape[0] - 1

    #YYYYYYYYYY
    ymin = 0
    while  ymin < img.shape[1] and is_low_contrast(img[:,ymin ,:]):
        ymin += 1
    if ymin >= img.shape[1] - 1:
        return [-1,-1,-1,-1,-1,-1]
    ymin -= 1
    if ymin<0:
        ymin=0

    ymax = img.shape[1] - 1
    while ymax > 0 and is_low_contrast(img[:,ymax ,: ]) :
        ymax -= 1
    if ymax <= 1:
        return [-1,-1,-1,-1,-1,-1]
    ymax += 1
    if ymax>img.shape[1]-1:
        ymax = img.shape[1] - 1

    #ZZZZZZZZZZ
    zmin = 0
    while zmin < img.shape[2] and is_low_contrast(img[:, :, zmin]) :
        zmin += 1
    if zmin >= img.shape[2] - 1:
        return [-1, -1, -1, -1, -1, -1]
    zmin -= 1
    if zmin<0:
        zmin=0

    zmax = img.shape[2] - 1
    while zmax > 0 and is_low_contrast(img[:, :, zmax]):
        zmax -= 1
    if zmax <= 1:
        return [-1, -1, -1, -1, -1, -1]
    zmax += 1
    if zmax>img.shape[2]-1:
        zmax = img.shape[2] - 1

    return [xmin,xmax,ymin,ymax,zmin,zmax]


def crop_and_resize(image, seg, img_size):
    area_to_crop = get_crop_3D(image)
    cropped_image = get_equal_crop_3D(image, area_to_crop)
    cropped_seg = get_equal_crop_3D(seg, area_to_crop)
    resized_image = resize(cropped_image, [img_size, img_size, img_size], preserve_range=True).astype(image.dtype)
    resized_seg = resize(cropped_seg, [img_size, img_size, img_size], preserve_range=True, anti_aliasing=False,
                         order=0).astype(seg.dtype)
    return resized_image, resized_seg


def decompress_and_format(filepath, filename):
    print('Decompressing image...')
    with gzip.open(filepath, 'rb') as f_in:
        with open(filename, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)
    image3D = io.imread(filename)
    os.remove(filename)
    print('Formating image...')
    (unique, counts) = np.unique(image3D, return_counts=True)
    idx = np.where(counts == counts.max())
    background = unique[idx][0]
    maxi = np.mean(image3D) + 5 * np.std(image3D)
    image3D = np.where(image3D < background, background, image3D)
    image3D = np.where(image3D > maxi, maxi, image3D)
    image3D = (image3D - background) * 255 / (maxi - background)
    print('Image is ready.')
    return image3D.astype(np.uint8)


def decompress(filepath, filename):
    print('Decompressing segmentation image...')
    with gzip.open(filepath, 'rb') as f_in:
        with open(filename, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)
    seg3D = io.imread(filename)
    os.remove(filename)
    print('Segmentation image is ready.')
    return seg3D.astype(np.uint8)


def decompress16bit(filepath, filename):
    print('Decompressing segmentation image...')
    with gzip.open(filepath, 'rb') as f_in:
        with open(filename, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)
    seg3D = io.imread(filename)
    os.remove(filename)
    print('Segmentation image is ready.')
    return seg3D.astype(np.float16)


def generator(list_data, output_path):
    start = time.time()
    list_errors = []
    while list_data:
        data = list_data[0]
        print(data)
        list_data.remove(data)
        seg_path, name = data
        # try:
        image3D = decompress_and_format(seg_path.replace('seg/', 'fuse/').replace('_seg_', '_fuse_'),
                                        seg_path.replace('seg/', 'fuse/').replace('_seg_', '_fuse_')[:-3])
        seg3D = decompress(seg_path, seg_path.replace('seg/', 'fuse/')[:-3])
        image3D, seg3D = crop_and_resize(image3D, seg3D, 256)
        # io.imsave(output_path + f"{name}.tiff", seg3D)
        io.imsave(output_path + 'images/{}.tiff'.format(name), image3D)
        io.imsave(output_path + 'segs/{}.tiff'.format(name), seg3D-1)
        # list_of_labels = np.unique(seg3D)[1:]
        # num_cells = list_of_labels.shape[0]
        # minimasks = np.zeros((56,56,56, num_cells), dtype=bool)
        # f = open(output_path + 'boxes/{}.dat'.format(name), 'w')
        # for c in range(num_cells):
        #     mask = np.where(seg3D == list_of_labels[c], 1, 0).astype(bool)
        #     horizontal_indicies = np.where(np.any(np.any(mask, axis=0), axis=1))[0]
        #     vertical_indicies = np.where(np.any(np.any(mask, axis=1), axis=1))[0]
        #     profound_indicies = np.where(np.any(np.any(mask, axis=0), axis=0))[0]
        #     x1, x2 = horizontal_indicies[[0, -1]]
        #     y1, y2 = vertical_indicies[[0, -1]]
        #     z1, z2 = profound_indicies[[0, -1]]
        #     f.write('{}\t{}\t{}\t{}\t{}\t{}\n'.format(y1 - 2, x1 - 2, z1 - 2, y2 + 2, x2 + 2, z2 + 2))
        #     minimasks[..., c] = resize(mask[y1 - 2:y2 + 2, x1 - 2:x2 + 2, z1 - 2:z2 + 2],
        #                                [56, 56, 56], preserve_range=True, anti_aliasing=False, order=0).astype(
        #         bool)
        # with bz2.BZ2File(output_path+'minimasks/{}.pickle'.format(name), 'w') as f:
        #     cPickle.dump(minimasks, f)
        # f.close()
        # except:
        #     print('erreur')
        #     list_errors.append(seg_path)
    print(list_errors)
    end = time.time()
    print(end-start)


def mask_generator(list_data, output_path):
    start = time.time()
    list_errors = []
    while list_data:
        seg_path = list_data[0]
        list_data.remove(seg_path)
        print(seg_path)
        # os.system('mkdir masks/{}'.format(seg_path.split('/')[-1][:-5]))
        # try:
        seg3D = io.imread(seg_path)
        list_of_labels = np.unique(seg3D)[1:]
        num_cells = list_of_labels.shape[0]
        # masks = np.zeros((256,256,256, num_cells), dtype=bool)
        arg_masks = []
        for c in range(num_cells):
            arg_masks.append(np.argwhere(seg3D == list_of_labels[c]).astype(np.uint8))
            # mask = np.where(seg3D == list_of_labels[c], 1, 0).astype(bool)
            # masks[..., c] = mask
        with bz2.BZ2File(output_path + '{}'.format(seg_path.split('/')[-1].replace('.tiff', '.pickle')), 'w') as f:
            cPickle.dump(arg_masks, f)
        # with bz2.BZ2File(output_path+'{}'.format(seg_path.split('/')[-1].replace('.tiff', 'b.pickle')), 'w') as f:
        #     cPickle.dump(masks, f)
        # np.save(output_path+'{}'.format(seg_path.split('/')[-1].replace('.tiff', '.npy')), masks)
        # except:
        #     print('erreur')
        #     list_errors.append(seg_path)
    print(list_errors)
    end = time.time()
    print(end-start)


# dataset_dir = './'
# dataset_dir = '/media/emmanuelfaure/BIG10T/DATABASE/SegmentedMembrane/Embryos/SPIM-Phallusia-Mammillata/'
# dataset_dir = '/gpfsscratch/rech/umm/ukw72um/data/Embryos/'
# dataset_dir = '/home/gdavid/Projets/Embryo3D/segs/'
dataset_dir = "/home/gdavid/Projets/DATA/140317-Patrick-St8/seg/"

# list_embryos = ['201215-Ninho', '170728-Samson-St8', '170302-Carl-St8', '181219_Anastasia_ST5',
#                 '180926-Come-Nodal-St8', '190919-Emilie', '180411-Stanislas-St5', '170119-Marius-St8',
#                 '140317-Patrick-St8', '180824-Emilie-Nodal-St8', '190517-Pascal-U0126', '180910-Ines-St8',
#                 '170908-Adriana-St8', '160707-Ralph-St8', '200825-Louise', '190417-Evan-U0126', '190926-Gedeon',
#                 '170226-Alexandre-St8', '200819-Eudes', '170225-Nestor-St8', '170129-Gildas-St8']
# list_cut = [None, -4, -4, -4, -4, None, -4, -4, -4, -4, -6, -4, -4, -4, None, -6, None, -4, None, -4, -4]

list_embryos = ['140317-Patrick-St8']
list_cut = [-4]

output_path = "/home/gdavid/Bureau/Patrick/"
# output_path = "/home/gdavid/Projets/NEMBRYOS/SEGMENTATION_COMPARISON/GT/"
# output_path = "/home/gdavid/Projets/DATA/Patrick_a00/"
# output_path = '/home/gdavid/Projets/Embryo3D/masks/'
# output_path = '/media/emmanuelfaure/BIG10T/Embryos3D/'
# output_path = '/gpfsscratch/rech/umm/ukw72um/data/Embryo3D/256/'

# list_segs = ["140317-Patrick-St8_seg_t050.nii.gz", "140317-Patrick-St8_seg_t001.nii.gz"]
list_segs = [f"140317-Patrick-St8_seg_t{str(i).zfill(3)}.nii.gz" for i in range(1, 11)]
list_segs = [dataset_dir + seg for seg in list_segs]
# list_segs = os.listdir(dataset_dir)
# list_segs.remove('Anastasia_t055.tiff')
# list_segs.remove('Anastasia_t056.tiff')
# for dirpath, _, filenames in os.walk(dataset_dir):
#     for filename in filenames:
#         list_segs.append(os.path.abspath(os.path.join(dirpath, filename)))
# list_segs = list(filter(lambda x: '_seg_' in x and not 'Giuletta' in x, list_segs))
#
# list_segs = list(filter(lambda x: 'Anastasia_ST5_seg_t055' in x or 'Anastasia_ST5_seg_t056' in x, list_segs))

list_data = []
for i, embryo in enumerate(list_embryos):
    for seg in list_segs:
        if embryo in seg:
            time_step = seg[-11:-7]
            if embryo == '180824-Emilie-Nodal-St8':
                embryo_name = 'Eminie'
            else:
                embryo_name = embryo[7: list_cut[i]]
            name = embryo_name + '_' + time_step
            list_data.append([seg, name])
# list_data = [dataset_dir + seg for seg in list_segs]
# list_data = list_data[:10]
# list_data = ['/home/gdavid/Projets/Embryo3D/segs/Patrick_t189.tiff']


nb_thread = 5
for i in range(nb_thread):
    x = threading.Thread(target=generator, args=(list_data, output_path))
    x.start()

