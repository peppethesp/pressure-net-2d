Create the dataset
====
This folder has been created to hos all the scripts and functions needed to create the dataset data that is used to train later the Neural Network.

The structure is this:
* in "create_dataset.py" are present all the functions required to build the tensorflow dataset required; these generate and return it according to the (input, label) format
* "execute.py" is mainly a script with the task of building and saving the dataset in chunks for the training