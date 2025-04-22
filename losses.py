import tensorflow as tf

def charbonnier_loss(y_true, y_pred, epsilon=1e-3):
    diff = y_true - y_pred
    loss = tf.sqrt(tf.square(diff) + epsilon**2)
    return tf.reduce_mean(loss)