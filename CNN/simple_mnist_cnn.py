from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt
import tensorflow as tf

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

X = tf.placeholder(tf.float32, [None, 784])
X_img = tf.reshape(X, [-1,28,28,1])

Y = tf.placeholder(tf.float32, [None, 10])

# L1 imgIn shape

W1 = tf.Variable(tf.random_normal([3,3,1,32],stddev=0.01))

# conv => (?, 28, 28, 32 )
# pool => (?, 14, 14, 32 )
L1 = tf.nn.conv2d(X_img, W1, strides=[1,1,1,1], padding='SAME')
L1 = tf.nn.relu(L1)
L1 = tf.nn.max_pool(L1, ksize=[1,1,1,1], strides=[1,2,2,1], padding='SAME')


# L2 imgIn shape ( ?, 14, 14, 32 )

# 다음 가중치 전 레이어의 필터갯수 32 , 다음 필터갯수 64로 변동
W2 = tf.Variable(tf.random_normal([3,3,32,64], stddev=0.01))

# conv => (?, 14, 14, 64)
# pool => (?, 7, 7, 64)
L2 = tf.nn.conv2d(L1, W2, strides=[1,1,1,1], padding='SAME')
L2 = tf.nn.relu(L2)
L2 = tf.nn.max_pool( L2, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

# fully connected layer로 변경 (FC 로 변경)
# 원래 예전에 784개 에서 3136 으로 변경되게 된다
L2 = tf.reshape(L2, [-1,7 * 7 * 64])

# 마지막으로는 10개의 output으로 classification을 하면된다
W3 = tf.get_variable("W3", shape=[7 * 7 * 64, 10], initializer=tf.contrib.layers.xavier_initializer())
b = tf.Variable(tf.random_normal([10]))

hypothesis = tf.matmul(L2, W3) + b

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits( logits=hypothesis, labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate=0.01).minimize(cost)


sess = tf.Session()

sess.run(tf.global_variables_initializer())

trainig_epoch = 2
batch_size = 100

#train my model
for epoch in range(trainig_epoch):
    avg_cost = 0
    total_batch = int(mnist.train.num_examples / batch_size)
    for i in range(total_batch):
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        feed_dict = {X : batch_xs, Y : batch_ys}
        c, _ = sess.run([cost, optimizer], feed_dict=feed_dict)
        avg_cost += c / total_batch
    print('Epoch:','%04d' % (epoch + 1), 'cost =', '{:.9f}'.format(avg_cost))

print('Learning Finished!')

# Test model check accuracy
correct_prediction = tf.equal(tf.argmax(hypothesis, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print('Accuracy:', sess.run(accuracy, feed_dict={X:mnist.test.images, Y:mnist.test.labels}))







