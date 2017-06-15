import numpy as np
import tensorflow as tf

image = np.array([[[[4], [3]],
                   [[2], [1]]]], dtype=np.float32)

with tf.Session() as sess:


    pool = tf.nn.max_pool(image, ksize=[1,2,2,1],
                          strides=[1,1,1,1], padding='SAME')

    print(pool.shape)
    print(pool.eval())

