# Project_3-Classification-of-patient-conditions

Providing multi-classification of 8 classes models of patient conditions using LightGBM, KNN, DecisionTree, and DQN for imbalanced classification.


## Data

- The data consist of 68-features and 8-classes.
- A number of data points are 622.
- A format is '.csv'.
- So many missing values that '.fillna(0)' used for imputation.
- A number of classes are imbalanced.


## Models

- Tree-based algorithms
  - LightGBM
  - DecisionTree
  
- The others
  - KNN
  - DQN for imbalanced classification (Lin, Enlu, Qiong Chen, and Xiaoming Qi. "Deep reinforcement learning for imbalanced classification." arXiv preprint arXiv:1901.01379 (2019).)


## Validation method

- 5-Cross validation was used.
- The whole data was split into 5-validation sets(train/test) in which the number of each class were considered. 
- Each validation sets ware saved as .csv format.


## Some issues dealt with

- A big different performance in OS (Win10 vs. Ubuntu 18.04)
  - About 20%p accuracy difference.

- As too many features(68) in the data, it is important to select the most effective set of features (roughly brute force). 

- Drawing out visualization for results of each algorithms.
  - Graphviz for DecisionTree
  - plot for LightGBM(https://github.com/microsoft/LightGBM/tree/master/examples/python-guide).
  - plot for KNN.
   
