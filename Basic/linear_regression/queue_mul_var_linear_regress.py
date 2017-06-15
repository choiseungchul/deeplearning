import tensorflow as tf
# import numpy as np


file_name_queue = tf.train.string_input_producer(
    ['data-01-test-score.csv', 'data-02-test-score.csv', 'data-03-test-score.csv'],
    shuffle=False, name='filename_queue')

reader = tf.TextLineReader()

key, value = reader.read(file_name_queue)

record_default = [[0.],[0.],[0.],[0.]]

xy = tf.decode_csv(value, record_defaults=record_default)

X = tf.placeholder(tf.float32, shape=[None, 3])
Y = tf.placeholder(tf.float32, shape=[None, 1])

W = tf.Variable(tf.random_normal([3,1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')

hypothesis = tf.matmul(X, W) + b

cost = tf.reduce_mean( tf.square(hypothesis - Y) )

optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-5)
train = optimizer.minimize(cost)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

train_x_batch, train_y_batch = \
    tf.train.batch( [xy[0:-1], xy[-1:]], batch_size=10 )

sess = tf.Session()

coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(sess=sess, coord=coord)

for step in range(2001):
    x_batch, y_batch = sess.run([train_x_batch, train_y_batch])

    # if step % 20 == 0:
    #     print(step, "Cost:", cost_val, "\nPrediction:\n", hy_val)

coord.request_stop()
coord.join(threads)