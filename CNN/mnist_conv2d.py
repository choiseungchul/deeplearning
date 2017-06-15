from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt
import tensorflow as tf

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

img = mnist.train.images[0].reshape(28,28)

plt.imshow(img, cmap='gray')
plt.show()


sess = tf.InteractiveSession()

img = img.reshape(-1, 28, 28, 1)

# filter 3x3x1 5개
# stride 2x2 으로 한다
# padding SAME 이므로
# 따라서 output은 (29 - 3) / 2 + 1 => 14x14
W1 = tf.Variable(tf.random_normal([3,3,1,5], stddev=0.01))
conv2d = tf.nn.conv2d(img, W1, strides=[1,2,2,1], padding='SAME')

# 14x14 conv이미지 생성됨
print(conv2d)

sess.run(tf.global_variables_initializer())

conv2d_img = conv2d.eval()

# max pooling
pool = tf.nn.max_pool(conv2d, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

print(pool)

sess.run(tf.global_variables_initializer())
pool_img = pool.eval()



