import tensorflow as tf
from tensorflow.keras import layers

class FRFN(tf.keras.layers.Layer):
    def __init__(self, dim=32, hidden_dim=128):
        super(FRFN, self).__init__()
        self.dim = dim
        self.hidden_dim = hidden_dim

        # Split dimension for partial convolution
        self.dim_conv = dim // 4
        self.dim_untouched = dim - self.dim_conv

        # Partial 3x3 conv block
        self.partial_conv = tf.keras.layers.Conv2D(
            filters=self.dim_conv,
            kernel_size=3,
            padding='same',
            use_bias=False,
        )

        self.x2_proj_conv = tf.keras.layers.Conv2D(
            filters=self.dim_conv,
            kernel_size=1,
            padding='same'
        )

        # Dense layers for gate mechanism
        self.linear1 = layers.Dense(hidden_dim * 2, activation='gelu')
        self.dwconv = layers.DepthwiseConv2D(kernel_size=3, padding='same', activation='gelu')
        self.linear2 = layers.Dense(dim)

    def call(self, x):
        B, H, W, C = tf.shape(x)[0], tf.shape(x)[1], tf.shape(x)[2], tf.shape(x)[3]

        # Partial convolution on the channel-split image
        x1, x2 = tf.split(x, [self.dim_conv, self.dim_untouched], -1)
        x1 = self.partial_conv(x1)

        # Project x2 to match and add it to x1
        # (i.e. simplification of concat which gave us issues when debugging)
        x2_proj = self.x2_proj_conv(x2)
        x = x1 + x2_proj

        # Linear layer
        x = tf.reshape(x, [B, H * W, self.dim_conv])
        x_gated = self.linear1(x)
        x1, x2 = tf.split(x_gated, 2, -1)

        # Spatial restore and depthwise conv on x1
        x1 = tf.reshape(x1, [B, H, W, self.hidden_dim])
        x1 = self.dwconv(x1)
        x1 = tf.reshape(x1, [B, H * W, self.hidden_dim])

        x_out = x1 * x2
        x_out = self.linear2(x_out)
        x_out = tf.reshape(x_out, [B, H, W, self.dim])
        return x_out