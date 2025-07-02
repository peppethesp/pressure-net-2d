# Add to path to get all the user defined packages in the project
import sys
import os
sys.path.insert(0, os.getcwd())
import tensorflow as tf
import numpy as np

from myPackageUtils.spatial_filter import spatial_filter
import random

# Files that contains implementaions of different approaches to generate random datasets


"""
Example:
    d_size = 250
    (N_x, N_y) = (128,128)
    ds = create_dataset_3(d_size, N_x, N_y)
"""
def create_dataset_1(d_size : int, N_x : int, N_y : int):
    """
        - d_size: Size of the dataset
        - N_x: Spatial dimension on the x-axis
        - N_y: Spatial dimension on the y-axis
        Return dataset with the previously described characteristics
        
        It assumes to feed in as input the divergence of a randomly generated velocity vector field.

        (inputs, labels) -> (image, image)
        """
    features = np.zeros((d_size, N_x, N_y, 1))
    for i in range(d_size):
        # define the velocity field
        stddev = 0.3
        u_t = tf.random.normal(shape=(1,N_x,N_y,1), stddev=stddev)
        v_t = tf.random.normal(shape=(1,N_x,N_y,1), stddev=stddev)

        # Define derivatives templates
        D_x = 0.5 * tf.constant(
            [[0., -1., 0.],
            [0., 0., 0.],
            [0., 1., 0.]]
        )
        D_x = tf.reshape(D_x, [3,3,1,1])
        D_y = 0.5 * tf.constant(
            [[0., 0., 0.],
            [-1., 0., 1.],
            [0., 0., 0.]]
        )
        D_y = tf.reshape(D_y, [3,3,1,1])

        # Divergence of velocity field
        u_x = tf.nn.convolution(u_t, D_x, padding="SAME")
        u_y = tf.nn.convolution(v_t, D_y, padding="SAME")
        div_u = u_x + u_y

        # print(div_u.numpy().shape)
        features[i,:,:,:] = div_u.numpy()[0,:,:,:]
        print(i, end="\r")

        # In my case the labels and the features coincide
    # --------- BUILDING DATASET --------- #
    dataset = tf.data.Dataset.from_tensor_slices((features, features))
    # --------- BUILDING DATASET --------- #
    return dataset

# -------

def create_dataset_2(d_size : int, N_x : int, N_y : int):
    """
        - d_size: Size of the dataset
        - N_x: Spatial dimension on the x-axis
        - N_y: Spatial dimension on the y-axis
        Return dataset with the previously described characteristics
        
        It assumes to feed in as input the laplacian of a randomly generated velocity vector field.

        (inputs, labels) -> (image, image)
    """

    # Laplacian filter
    L = tf.constant(
        [[0, 1, 0],
        [1, -4, 1],
        [0, 1, 0]], dtype=tf.float32)
    L = tf.reshape(L, [3,3,1,1])
    
    # Let us create the needed dataset:(inputs, labels)
    inputs = np.zeros((d_size, N_x, N_y, 1))
    for i in range(d_size):
        # define the velocity field
        stddev = 1/20
        p_true = tf.random.normal(shape=(1,N_x,N_y,1), stddev=stddev)

        # Divergence of velocity field
        div_u = tf.nn.convolution(p_true, L, padding="SAME")

        # print(div_u.numpy().shape)
        inputs[i,:,:,:] = div_u.numpy()[0,:,:,:]
        print(i, end="\r")

    # In my case the labels and the features coincide
    # --------- BUILDING DATASET --------- #
    dataset = tf.data.Dataset.from_tensor_slices((inputs, inputs))
    # --------- BUILDING DATASET --------- #
    return dataset

# -------


def create_dataset_spatial_filtering(d_size : int, N_x : int, N_y : int, filtering_radius : int):
    """
        - d_size: Size of the dataset
        - N_x: Spatial dimension on the x-axis
        - N_y: Spatial dimension on the y-axis
        - filtering_radius: Spatial dimension on the y-axis

        Return dataset with the previously defined parameters; a spatial filter is applied to remove high frequency variations from the dataset; this is done so that the dataset has input with different variabilities ant so that it can better discover the underlying relationship.
    """
    x = np.arange(N_x)
    y = np.arange(N_y)

    # Laplacian filter
    L = tf.constant(
        [[0, 1, 0],
        [1, -4, 1],
        [0, 1, 0]], dtype=tf.float64)
    L = tf.reshape(L, [3,3,1,1])

    # Let us create the needed dataset:(features, labels)
    features = np.zeros((d_size, N_x, N_y, 1))
    for i in range(d_size):
        # define the velocity field
        random_noise1 = np.random.normal(size=(N_x, N_y))
        p_true = spatial_filter(random_noise1, char_len=random.uniform(0,filtering_radius)).astype(np.double)
        p_true = tf.constant(p_true.reshape((1,N_x, N_y,1)), dtype=tf.float64)

        # Divergence of velocity field - Normalization
        div_u = tf.nn.convolution(p_true, L, padding="SAME")
        div_std = tf.math.reduce_std(div_u)
        div_u = 0.2*div_u/div_std

        # print(div_u.numpy().shape)
        features[i,:,:,:] = div_u.numpy()[0,:,:,:]
        print(i, end="\r")

    # In my case the labels and the features coincide
    # --------- BUILDING DATASET --------- #
    dataset = tf.data.Dataset.from_tensor_slices((features, features) )
    # --------- BUILDING DATASET --------- #
    return dataset