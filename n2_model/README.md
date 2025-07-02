# Custom model
In this folder are present all the script to define the custom keras/tensorflow implementation of the neural network.

To overcome the keras limitation that requires the loss function to accept as parameters:
    (y_true, y_pred)
the custom loss function is based on the assumption that the dataset has the structure
    (input, input)
so that the custom loss can compute the laplacian of the output y_pred and compare it with the input.

This solution is, evidently, sub-optimal and redundant. Future implementations will fix this issue by subclassing the keras loss class.