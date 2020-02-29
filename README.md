# Project_3-Classification-of-patient-conditions

Providing multi-classification of 8 classes models of patient conditions using LightGBM, KNN, DecisionTree, and DQN for imbalanced classification.


## Data

- The data consist of 68-features and 8-classes.
- A number of data points are 622.
- A format is '.csv'.
- So many missing values that '.fillna(0)' used for imputation.
- A number of classes are imbalanced.


## Validation method

- 5-Cross validation was used.
- The whole data was split into 5-validation sets(train/test) in which a ratio of each class were considered. 


## Some issues dealt with

- Drawing out visualization for results of each algorithms.
  - Graphviz for DecisionTree
  - plot for LightGBM(https://github.com/microsoft/LightGBM/tree/master/examples/python-guide)
  - plot for KNN
    
