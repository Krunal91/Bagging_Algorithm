# Bagging Machine Learning Algorithm in Python
In this Bagging algorithm, I am using decision stump as a weak learner.  

**Decision Stump :** It means decision tree which has depth of 1.

## Files and Data Descriptions

#### 1. decision_tree.py  
This is main python file.To run this project, one just have to run this files. All the function calls to different files will be made from this main python file.

#### 2. dataset_details.py  
This function loads the data from input data folder and process the data to make it appropriate for the decision tree algorithms.

#### 3. information_gain.py  
This file contains the implementation of splitting criteria named information gain.

#### 4. prediction.py  
This file uses the already built structure to predit the class for new data.  

#### 5. data  
This folder contains the input data.

#### bagging.py
This file does the necessary changes so that data other functions runs smoothly.


## How to run the code  

#### Script requirement  
Scirpt should have read, write and delete permission

#### Script running instruction
Run **main.py** script in command line as below

python **main.py** <ensemple_type> <treeDeoth> <bags> <data folder path>

example: python main.py bag 5 10 "C:/Users/data/"

ensemple_type = bag ; for nagging
	      = boost; for boosting
