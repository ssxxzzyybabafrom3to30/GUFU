# GUFU

There are several files for running the model.

- ```util.py``` provides utility functions for GUFU.
- ```graph.py``` provides the weighted graph formulations using network.
- ```prep.py``` generates output folder from the input data, which is the data sampled from one of our mall test sites. This file has to be run before other files.
- ```ae.py``` refers to the feature extractor. You may test its performance via an independent main function in it.
- ```sage_ve.py``` provides the graph aggregation unit, i.e. for graph initialization and update. Try running one iteration of update using the independent main function in it.
- ```lp.py``` is for link prediction. From a graph's weighted edgelist, it follows our paper's design to connect new edges and output the new edgeslist.
- ```pred.py``` is for automatic training and updating. Number of weeks is a hyperparameter (in our example 5) to set the number of batches of update we are expecting.

Hyperparameters mentioned in the paper are passed as default values of the models in ```ae.py```, ```sage_ve.py``` and ```lp.py```.
