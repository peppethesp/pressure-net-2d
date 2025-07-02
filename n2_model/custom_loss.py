import tensorflow as tf
import keras
# Laplacian filter
L = tf.constant(
    [[0, 1, 0],
    [1, -4, 1],
    [0, 1, 0]], dtype=tf.float32)
L = tf.reshape(L, [3,3,1,1])

# Definition of the LOSS
def f_obj(rho, W):
    @keras.saving.register_keras_serializable()
    def loss(div_u, p_t):
        # laplacian of pressure
        lap_p = tf.nn.convolution(p_t, L, padding="SAME")

        # Computing objective
        diff = tf.math.squared_difference(div_u, 1/rho * lap_p)
        # obj = tf.math.reduce_sum(tf.math.multiply(W, diff))
        obj = tf.math.reduce_sum(diff)
        return obj
    return loss