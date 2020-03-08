# Project_3-Multi-classification-of-patient-conditions

Providing multi-classification of 8 classes models of patient conditions using LightGBM.

## Data

- The data consist of 68-features and 8-classes.
- A number of data points are 622.
- A format is '.csv'.
- So many missing values that '.fillna(0)' used for imputation.
- A number of classes are imbalanced.


## Models

-LightGBM


## Validation method

- Leave One Out cross validation which is for small data-set.
- 5-fold cross validation.


## Some issues dealt with

- A big different performance in OS (Win10 vs. Ubuntu 18.04)
  - About 20%p accuracy difference.

- As too many features(68) in the data, it is important to select the most effective set of features (roughly brute force). 

- Drawing out visualization for results of each algorithms.
  - Graphviz for DecisionTree
  - Graphviz ans plotting for LightGBM(https://github.com/microsoft/LightGBM/tree/master/examples/python-guide).
  - t-SNE for KNN (t-SNE is visualization mothed for high dimension data in which features of data are reduced to lower dimension).
  
- Plotting learning curves of each algorithms.
