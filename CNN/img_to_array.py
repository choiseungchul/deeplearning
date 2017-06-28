import tensorflow as tf
import numpy as np
import pprint as pp


filenames = ['./temp/img2.jpg']
filename_queue = tf.train.string_input_producer(filenames)

reader = tf.WholeFileReader()
key, value = reader.read(filename_queue)

images = tf.image.decode_jpeg(value, channels=3)


init_op = tf.global_variables_initializer()

with tf.Session() as sess:
  sess.run(init_op)

  for i in range(1): #length of your filename list
    image = images.eval() #here is your image Tensor :)


  pp.pprint(image)







