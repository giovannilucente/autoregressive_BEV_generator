# Autoregressive Occupancy Prediction
This repository contrins the scripts to train an autoregressive model to predict traffic occupancy. The model is trained on the Waymo motion dataset.

## Copy the repository
``` bash
git clone https://github.com/lucegi/waymo_autoregressive_occupancy_prediction.git
cd waymo_autoregressive_occupancy_prediction
```
## Download the Dataset
``` bash
chmod +x download_waymo_dataset.sh
bash download_waymo_dataset.sh
```
The dataset will have the following structure:
```bash
/scenario/
├── training/
│   ├── training_tfexample.tfrecord-00000-of-01000
│   ├── training_tfexample.tfrecord-00001-of-01000
│   ├── ...
├── validation/
│   ├── validation_tfexample.tfrecord-00000-of-01000
│   ├── validation_tfexample.tfrecord-00001-of-01000
│   ├── ...
├── testing/
│   ├── testing_tfexample.tfrecord-00000-of-00150
│   ├── ...
```


## Convert the Dataset
```bash
python3 convert_dataset.py
``` 
The converted dataset will have the following structure:
```bash
/converted_dataset/
├── train/
│   ├── training_tfexample-00000-of-01000
│   │   ├── 1
│   │   │   ├── win000.png
│   │   │   ├── win010.png
│   │   │   ├── win020.png
│   │   │   ├── win030.png
│   │   │   ├── win040.png
│   │   │   ├── win050.png
│   │   │   ├── win060.png
│   │   │   ├── win070.png
│   │   ├── 2
│   │   ├── ...
│   ├── training_tfexample-00001-of-01000
│   ├── ...
├── validation/
│   ├── validation_tfexample-00000-of-01000
│   ├── validation_tfexample-00001-of-01000
│   ├── ...
├── test/
│   ├── testing_tfexample-00000-of-00150
│   ├── ...
```
With the images ```win0XX.png``` that look like:

![Alt text](media/win030.png)

## Training
To train the model run:
```bash
python3 train.py
```
