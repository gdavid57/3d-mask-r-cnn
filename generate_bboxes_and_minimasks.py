import os
import numpy as np
from skimage import io
import bz2
import _pickle as cPickle
from skimage.transform import resize
import threading
import time


def generate_phallusia_example(list_seg_names, astec_ground_truth_path, output_path):

    start = time.time()
    list_errors = []

    while list_seg_names:

        seg_name = list_seg_names[0]
        list_seg_names.remove(seg_name)
        print(seg_name)
        seg_path = f"{astec_ground_truth_path}{seg_name}"

        try:

            box_name = seg_name.split(".")[0].replace("ASTEC", "boxes")
            minimask_name = seg_name.split(".")[0].replace("ASTEC", "minimasks")

            seg3D = io.imread(seg_path)

            list_of_labels = np.unique(seg3D)[1:]
            num_cells = list_of_labels.shape[0]

            minimasks = np.zeros((56,56,56, num_cells), dtype=bool)
            f = open(output_path + 'Boxes/{}.dat'.format(box_name), 'w')

            for c in range(num_cells):

                mask = np.where(seg3D == list_of_labels[c], 1, 0).astype(bool)

                horizontal_indicies = np.where(np.any(np.any(mask, axis=0), axis=1))[0]
                vertical_indicies = np.where(np.any(np.any(mask, axis=1), axis=1))[0]
                profound_indicies = np.where(np.any(np.any(mask, axis=0), axis=0))[0]
                x1, x2 = horizontal_indicies[[0, -1]]
                y1, y2 = vertical_indicies[[0, -1]]
                z1, z2 = profound_indicies[[0, -1]]

                f.write('{}\t{}\t{}\t{}\t{}\t{}\n'.format(y1 - 2, x1 - 2, z1 - 2, y2 + 2, x2 + 2, z2 + 2))

                minimasks[..., c] = resize(mask[y1 - 2:y2 + 2, x1 - 2:x2 + 2, z1 - 2:z2 + 2],
                                        [56, 56, 56], preserve_range=True, anti_aliasing=False, order=0).astype(
                    bool)
                
            with bz2.BZ2File(output_path+'Minimasks/{}.pickle'.format(minimask_name), 'w') as f:

                cPickle.dump(minimasks, f)

            f.close()

        except:

            print('erreur')
            list_errors.append(seg_path)
            
    print(list_errors)
    end = time.time()
    print(end-start)


astec_ground_truth_path = "data/ASTEC_Ground_truth/"

output_path = "data/"

os.makedirs(f"{output_path}Boxes/", exist_ok=True)
os.makedirs(f"{output_path}Minimasks/", exist_ok=True)

list_seg_names = os.listdir(astec_ground_truth_path)

nb_thread = 5
for i in range(nb_thread):
    x = threading.Thread(target=generate_phallusia_example, args=(list_seg_names, astec_ground_truth_path, output_path))
    x.start()

