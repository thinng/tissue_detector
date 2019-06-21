# Resources:
+ README.md: this file.
+ data1.7z and data2.7z: zipped data files. 
###  source codes:
+ train_all.py: running anomaly detection using all normal data for training.
+ train_10_to_90.py: running anomaly detection using 10,20,..,90 percent of normal data for training.

### folder data/:
+ data files extracted from data1.7z and data2.7z

### folder figs/:
+ visualization of normal versus tumor data points detected by the proposed method.

### folder results/:
Performance of the proposed method on different cancers, using either all or parts of normal data for training. Measured in F1, Precision, Recall, Specificity, Accuracy, and AUC.


# Step-by-step running:

## 0. Installing Python libaries needed
+ Install sklearn: pip install scikit-learn

## 1. Using all normal data for training
Running
```sh
python train_all.py
```
This returns results/result.csv, containing the performance of the proposed model on different cancers, using all normal data for training.
This code also returns a visualization of normal versus tumor data points detected by the proposed method.

## 2. Using parts of normal data for training
Running 
```sh
python train_10_to_90.py
```
This returns results/result_x0.csv, where x ranges from 1 to 9, containing the performance of the proposed model on different cancers, using 10,20,..,90 percent of normal data for training.

# Cite:

If you use this code in your work please cite our paper as follows:
```sh
@article{quinn2018cancer,
  title={Cancer as a tissue anomaly: classifying tumor transcriptomes based only on healthy data},
  author={Quinn, Thomas and Nguyen, Thin and Lee, Samuel and Venkatesh, Svetha},
  journal={Frontiers in Genetics},
  year={2019}
}
```