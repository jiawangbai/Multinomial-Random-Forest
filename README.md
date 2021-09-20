# Multinomial-Random-Forest
A python 3.x implementation of Multinomial Random Forests (MRF).

## Requirements
This package is developed with Python 3.x, please make sure all the dependencies are installed, 
which is specified in requirements.txt. Please run the following command to install dependencies before running the code:
> pip install -r requirements.txt.


## Description of files

|    \<file name\>     |            \<description\>                    |
|--------------------|---------------------------------------------|
|   data/car.data    | Data used in the demo.                      |
|      demo.py       | Example of MRF training and test.           |
|   DecisionNode.py  | Implementation of node that makes up the tree.    | 
|      Tree.py       | Implementation of tree classifier.               |
|  MultinomialRF.py  | Implementation of MRF classifier.                |
|     utils.py       | Utils used in the above codes.              |
 
## Run the demo

To run the demo, simply run demo.py by default parameters. 

> python3 demo.py

Modify the parameter CROSS_VALIDATION = True to run the cross validation.
