import tensorflow as tf
from models.ast_tf import ASTModel

model = ASTModel(embed_dim=16)

_ = model(tf.random.uniform([1, 64, 64, 3]), training=False)

ckpt = tf.train.Checkpoint(model=model)
status = ckpt.restore('train_v2_checkpoints_ast/ckpt-4')
status.expect_partial()

img_raw = tf.io.read_file('DID-MDN-split/input/3.png')

img = tf.image.decode_image(img_raw, channels=3)
img = tf.image.resize(img, [64,64])
img = tf.image.convert_image_dtype(img, tf.float32)

inp = tf.expand_dims(img, axis=0)  


out = model(inp, training=False)

arr = tf.clip_by_value(out[0], 0., 1.).numpy()*255
tf.keras.preprocessing.image.save_img('pred.png', arr.astype('uint8'))
