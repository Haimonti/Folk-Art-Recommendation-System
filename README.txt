# HSAL-GNN

##Code introduction
The code is implemented based on DGSR [Dynamic Graph Neural Networks for Sequential Recommendation](https://ieeexplore.ieee.org/abstract/document/9714053).

## Usage

### Generate data

You need to run the file ```new_data.py``` to generate the data format needed for our model. The detailed commands 
can be found in ```load_{dataset}.sh```

You need to run the file ```generate_neg.py``` to generate data to speed up the test.

You need to run the file ```series.py``` or ```rotary.py``` to generate positional embeddings.

### Training and Testing 

Then you can run the file ```new_main.py``` to train and test our model. 


## Requirements

- Python 3.12.7
- torch 2.4.0+cu121
- dgl 2.4.0+cu121
