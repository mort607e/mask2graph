# mask2graph

**mask2graph** is a Python library for converting **binary segmentation masks** into **graph-based representations**.

The project focuses on extracting structured, topologically meaningful graphs from raster masks, enabling downstream analysis such as connectivity assessment, network statistics, and vector export.

This repository was initially created to support my thesis. Alongside this repository a secondary repository was made to support the riverseg project. It is focused on extracting river extents from satellite imagery using deep learning techniques. It can be found here: [riverseg](https://github.com/mort607e/riverseg)


## Test data
This repository was initially made for large river masks, these files are quite large and as a result are not included in the repo. However they can be downloaded from here: [https://drive.google.com/file/d/1abPzbC0Mhuj2_Ra5qz0ydEbuGbobuqFh/view?usp=sharing](https://drive.google.com/file/d/1abPzbC0Mhuj2_Ra5qz0ydEbuGbobuqFh/view?usp=sharing)\
This zip file contains 3 large river masks in tif format. Place them in `data/samples` after downloading and unzipping, to use them with the provided examples. 
