# Multinomial-Random-Forest
This is the official implementation of our paper [Multinomial Random Forest](https://www.sciencedirect.com/science/article/pii/S0031320321005112), accepted by the Pattern Recognition (2021). 

## Requirements
This package is developed with Python 3.x, please make sure all the dependencies are installed, 
which is specified in requirements.txt. Please run the following command to install dependencies before running the code:
> pip install -r requirements.txt.


## Description of files

|    \<file name\>     |            \<description\>                    |
|--------------------|---------------------------------------------|
|   data/car.data    | Data using in the demo.                     |
|      demo.py       | Example of training and test MRF.           |
|   DecisionNode.py  | Implement of node that make up the tree.    | 
|      Tree.py       | Implement of tree classifier.               |
|  MultinomialRF.py  | Implement of MRF classifier.                |
|     utils.py       | Utils using in the above codes.             |
 
## Run the demo

To run the demo (with default parameters), simply run demo.py by

> python3 demo.py

Modify the parameter CROSS_VALIDATION = True to run the cross validation.

## Reference
If our work or this repo is useful for your research, please cite our paper as follows:

```
@article{bai2021multinomial,
  title={Multinomial Random Forest},
  author={Bai, Jiawang and Li, Yiming and Li, Jiawei and Yang, Xue and Jiang, Yong and Xia, Shu-Tao},
  journal={Pattern Recognition},
  pages={108331},
  year={2021},
  publisher={Elsevier}
}
```
