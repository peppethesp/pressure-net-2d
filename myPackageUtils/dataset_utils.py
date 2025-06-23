import os
import shutil
import tensorflow as tf

# Save the dataset to a file for later use
def save_dataset(dataset, path, overwrite=False):
    """ This function is used to save the dataset

        - "dataset": is the dataset
        - "path": is the absolute or relative path where to save the dataset
        - "overwrite": specifies if the dataset has to be overwrittenm defaults to 'False'
    """
    if (overwrite == True) and (os.path.exists(path)):
        shutil.rmtree(path)
    dataset.save(path)

# Load the dataset data from a folder
def load_dataset(dataset_path):
    """ Function that loads all the datasets prevoiusly saved in a folder "dataseth_path"

    - "dataset_path": the path where to look for the the dataset saved in chunks
    """
    dataset_names = os.listdir(dataset_path)
    
    for i, ds_name in enumerate(dataset_names):
        # Reconstruct system path
        ds_path = (os.path.join(dataset_path, ds_name))
        print("Loading dataset from: ", ds_path)
        if i == 0:
            dataset = tf.data.Dataset.load(ds_path)
        else:
            # If dataset has the first element concatenate with subsequent
            dataset = dataset.concatenate(tf.data.Dataset.load(ds_path))
    return dataset
