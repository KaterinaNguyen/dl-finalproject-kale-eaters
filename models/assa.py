import tensorflow as tf
from tensorflow.keras import layers

class ASSA(tf.keras.layers.Layer):
    def __init__(self, dim, num_heads=4):
        super(ASSA, self).__init__()
        self.num_heads = num_heads
        self.scale = tf.constant((dim // num_heads) ** -0.5, dtype=tf.float32)

        # Query, Key, Value projections for both attention types
        self.qkv_dense = layers.Dense(dim * 3)
        self.qkv_sparse = layers.Dense(dim * 3)

        self.proj_dense = layers.Dense(dim)
        self.proj_sparse = layers.Dense(dim)

        # Learnable fusion weights
        self.fusion_weight = tf.Variable(initial_value=tf.ones((2,)), trainable=True)

        self.norm = layers.LayerNormalization()

    def split_heads(self, x, B, N):
        x = tf.reshape(x, [B, N, self.num_heads, -1])
        return tf.transpose(x, [0, 2, 1, 3])

    def scaled_dot_product_attention(self, q, k, v):
        attn_logits = tf.matmul(q, k, transpose_b=True) * self.scale
        attn_weights = tf.nn.softmax(attn_logits)
        return tf.matmul(attn_weights, v)

    def relu_squared_attention(self, q, k, v):
        relu_q = tf.square(tf.nn.relu(q))
        relu_k = tf.square(tf.nn.relu(k))
        attn_logits = tf.matmul(relu_q, relu_k, transpose_b=True) * self.scale
        return tf.matmul(attn_logits, v)

    def call(self, x):
        B = tf.shape(x)[0]
        H = tf.shape(x)[1]
        W = tf.shape(x)[2]
        C = tf.shape(x)[3]
        N = H * W
        x_flat = tf.reshape(x, [B, N, C])
        x_flat = self.norm(x_flat)

        # Dense branch
        qkv_d = self.qkv_dense(x_flat)
        q_d, k_d, v_d = tf.split(qkv_d, 3, -1)
        q_d = self.split_heads(q_d, B, N)
        k_d = self.split_heads(k_d, B, N)
        v_d = self.split_heads(v_d, B, N)
        out_d = self.scaled_dot_product_attention(q_d, k_d, v_d)
        out_d = tf.transpose(out_d, [0, 2, 1, 3])
        out_d = tf.reshape(out_d, [B, N, C])
        out_d = self.proj_dense(out_d)

        # Sparse branch
        qkv_s = self.qkv_sparse(x_flat)
        q_s, k_s, v_s = tf.split(qkv_s, 3, -1)
        q_s = self.split_heads(q_s, B, N)
        k_s = self.split_heads(k_s, B, N)
        v_s = self.split_heads(v_s, B, N)
        out_s = self.relu_squared_attention(q_s, k_s, v_s)
        out_s = tf.transpose(out_s, [0, 2, 1, 3])
        out_s = tf.reshape(out_s, [B, N, C])
        out_s = self.proj_sparse(out_s)

        # Adaptive fusion
        alpha = tf.nn.softmax(self.fusion_weight)
        out = alpha[0] * out_s + alpha[1] * out_d
        out = tf.reshape(out, [B, H, W, C])

        return out