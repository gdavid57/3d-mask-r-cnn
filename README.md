<h1 align="center"> 3D Mask R-CNN </h1>

Based on the [2D implementation](https://github.com/matterport/Mask_RCNN) by Matterport, Inc, [this update](https://github.com/ahmedfgad/Mask-RCNN-TF2) and [this fork](https://github.com/matterport/Mask_RCNN/pull/1611/files).

This 3D implementation was written by Gabriel David (LIRMM, Montpellier, France). Most of the code inherits from the MIT Licence edicted by Matterport, Inc (see core/LICENCE).

This repository is linked to the paper:

**G. David and E. Faure, End-to-end 3D instance segmentation of synthetic data and embryo microscopy images with a 3D Mask R-CNN, Front. Bioinform., 27 January 2025, Volume 4 - 2024 | [DOI link](https://doi.org/10.3389/fbinf.2024.1497539)**

We adopt a Docker approach to simplify the distribution and reproduction of our work. Running this 3D Mask R-CNN without our image is possible but requires to install TensorFlow sources and to compile the 3D Non Max Suppression and 3D Crop And Reisze custom operations by hand.

Go back to [Toy dataset branch](https://github.com/gdavid57/3d-mask-r-cnn/tree/main).

# Phallusia mammillata Dataset

This section aims to reproduce the results of the paper mentioned above on the Phallusia mammillata dataset. Please follow the default commands below.

Representations of pair ground truth instance segmentation and input image:

<p align="center">
    <img src="example/input_image.gif" alt="Input image" width="40%"><img src="example/segmentation.gif" alt="Instance segmentation" width="40%">
</p>

Clone this repository and access cloned directory with:

```
git clone https://github.com/gdavid57/3d-mask-r-cnn.git
cd 3d-mask-r-cnn
git checkout -b morphogenesis
```

See the main branch for extensive details.

## Data

Data are hosted on figshare: [here](https://figshare.com/articles/dataset/3D_Mask_R-CNN_data/26973085). The user is expected to download all these files and to unzip them in a "data" folder within the 3d-mask-r-cnn folder, while the weights folder must be copied at the same level as the data, configs and core folders:

    /configs
    /core
    /data
        /ASTEC_Ground_truth
            PM1_t001_ASTEC.tiff
            ...
        /Inputs
            PM1_t001_input.tiff
            ...
    /weights
        /rpn
            epoch_019.h5
        /heads
            epoch_017.h5

To perform evaluation, one must generate ground truth data (bounding boxes and minimasks). To build these additive ground truths, run:

```
docker run -it --gpus "0" --volume $PWD:/workspace gdavid57/3d-mask-r-cnn python -m generate_bboxes_and_minimasks.py
```

The test set, composed of the so-called PM1 series, which constitutes the most reliable series in this dataset, is created using:

```
docker run -it --gpus "0" --volume $PWD:/workspace gdavid57/3d-mask-r-cnn python -m generate_datasets.py
```

For transparency during the review process, we also delivered the code to generate the RPN targets and to augment 3D instance segmentation data, although we do not use them for evaluation.


## Mask R-CNN evaluation

One can evaluate the performance of the whole Mask R-CNN with the command line:

```
docker run -it --gpus "0" --volume $PWD:/workspace gdavid57/3d-mask-r-cnn python -m main --task "MRCNN_EVALUATION" --config_path "configs/mrcnn/embryo_mrcnn_config.json" --summary
```

where

+ --config: whole Mask R-CNN config. See *embryo_mrcnn_config.json* or *core/config.py* for more details.
+ --summary: if True, display the Mask R-CNN keras model summary, number of examples in the test dataset, as well as the given config.

By default, the predicted instance segmentation are saved under *data/results/* and results such as mAP, precision and recall are gathered in the *report.csv* in the same folder.


# Troubleshooting

+ A recent version of Docker must be installed. See [here](https://docs.docker.com/engine/install/) for Docker installation. Follow the post-installation instructions to add current user to docker group.

+ In case of custom op error, please compile the 3D Non Max Suppression and 3D Crop And Resize on your own computer following [this tutorial](https://github.com/gdavid57/3d-nms-car-custom-op). The generated wheel archive should then be placed in core/custom_op/ of this repo, and the image must be rebuilt with

```
docker build -t IMAGE_NAME .
```

by replacing IMAGE_NAME by the name of your choice. Finally, you can use all the previous commands by changing the gdavid57/3d-mask-r-cnn by your image name.

+ In case of other problem, please open an Issue.