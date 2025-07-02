import keras
from keras import layers

def make_model(input_shape):
    # Define input 
    inputs = keras.Input(shape=(input_shape))
    x = layers.Conv2D(8, 3, activation='relu', padding='same') (inputs)
    x_1 = layers.AveragePooling2D(2) (x)
    x_2 = layers.AveragePooling2D(2) (x_1)

    x = layers.Conv2D(8, 3, activation='relu', padding='same') (x)
    x_1 = layers.Conv2D(8, 3, activation='relu', padding='same') (x_1)
    x_2 = layers.Conv2D(8, 3, activation='relu', padding='same') (x_2)

    # Upscaling of the pooled layers
    x_1 = layers.Resizing(input_shape[0], input_shape[1]) (x_1)
    x_2 = layers.Resizing(input_shape[0], input_shape[1]) (x_2)

    # Sum resized layers
    x = layers.Add() ([x_1, x_2, x])

    # Last convolution
    x = layers.Conv2D(8, 1, activation='relu', padding='same') (x)

    outputs = layers.Conv2D(1, 1, activation=None, padding='same') (x)
    return keras.Model(inputs=inputs, outputs=outputs, name='MyModel')
