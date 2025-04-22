import tensorflow as tf
from tensorflow.keras import layers
from models.assa import ASSA
from models.frfn import FRFN


class Downsample(tf.keras.layers.Layer):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = layers.Conv2D(out_channels, kernel_size=4, strides=2, padding='same')

    def call(self, x):
        return self.conv(x)


class Upsample(tf.keras.layers.Layer):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.deconv = layers.Conv2DTranspose(out_channels, kernel_size=2, strides=2, padding='same')

    def call(self, x):
        return self.deconv(x)


class ConvLayer(tf.keras.layers.Layer):
    def __init__(self, dim):
        super().__init__()
        self.norm = layers.LayerNormalization()
        self.assa = ASSA(dim)
        self.frfn = FRFN(dim)

    def call(self, x):
        # Transformer block
        residual = x
        x = self.norm(x)

        x = self.assa(x)
        x = tf.clip_by_value(x, -5.0, 5.0)
        x = x + residual

        residual = x
        x = self.frfn(x)
        x = tf.clip_by_value(x, -5.0, 5.0)
        x = x + residual
        return x


class ASTModel(tf.keras.Model):
    def __init__(self, img_size=256, input_channels=3, embed_dim=64):
        super().__init__()

        # Input projection
        self.input_proj = layers.Conv2D(embed_dim, kernel_size=3, padding='same', activation='relu')
        self.pos_drop = layers.Dropout(0.1)

        # Encoder layers
        self.encoder1 = ConvLayer(embed_dim)
        self.down1 = Downsample(embed_dim, embed_dim * 2)

        self.encoder2 = ConvLayer(embed_dim * 2)
        self.down2 = Downsample(embed_dim * 2, embed_dim * 4)

        self.encoder3 = ConvLayer(embed_dim * 4)
        self.down3 = Downsample(embed_dim * 4, embed_dim * 8)

        self.encoder4 = ConvLayer(embed_dim * 8)
        self.down4 = Downsample(embed_dim * 8, embed_dim * 16)

        # ASSA-based transformer block (Bottleneck)
        self.bottleneck = ConvLayer(embed_dim * 16)

        # Decoder
        self.up4 = Upsample(embed_dim * 16, embed_dim * 8)
        self.decoder4 = ConvLayer(embed_dim * 16)

        self.up3 = Upsample(embed_dim * 8, embed_dim * 4)
        self.decoder3 = ConvLayer(embed_dim * 8)

        self.up2 = Upsample(embed_dim * 4, embed_dim * 2)
        self.decoder2 = ConvLayer(embed_dim * 4)

        self.up1 = Upsample(embed_dim * 2, embed_dim)
        self.decoder1 = ConvLayer(embed_dim * 2)

        # Output
        self.output_proj = layers.Conv2D(input_channels, kernel_size=3, padding='same')

    def call(self, x):
        # Input projection
        x_input = x
        x = self.input_proj(x)
        x = self.pos_drop(x)

        # Encoder
        enc1 = self.encoder1(x)
        x = self.down1(enc1)

        enc2 = self.encoder2(x)
        x = self.down2(enc2)

        enc3 = self.encoder3(x)
        x = self.down3(enc3)

        enc4 = self.encoder4(x)
        x = self.down4(enc4)

        # Bottleneck
        x = self.bottleneck(x)

        # Decoder
        x = self.up4(x)
        x = layers.Concatenate(axis=-1)([x, enc4])
        x = self.decoder4(x)

        x = self.up3(x)
        x = layers.Concatenate(axis=-1)([x, enc3])
        x = self.decoder3(x)

        x = self.up2(x)
        x = layers.Concatenate(axis=-1)([x, enc2])
        x = self.decoder2(x)

        x = self.up1(x)
        x = layers.Concatenate(axis=-1)([x, enc1])
        x = self.decoder1(x)

        # Output
        residual = self.output_proj(x)
        output = x_input + residual
        output = tf.clip_by_value(output, 0.0, 1.0)
        return output