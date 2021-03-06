import tensorflow as tf
import numpy as np

'''
xor연산 정리 샘플
A   B   X 
0   0   0
1   0   1
0   1   1
1   1   0
이런식의 연산이 xor연산임
'''

x_data = np.array([[0,0],[1,0],[0,1],[1,1]], dtype=np.float32)
y_data = np.array([[0],[1],[1],[0]], dtype=np.float32)


# placeholder를 쓰기위해 X, Y를 지정
X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)
W = tf.Variable(tf.random_normal([2, 1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')

hypo = tf.sigmoid(tf.matmul(X, W) + b)

cost = -tf.reduce_mean(Y * tf.log(hypo) + (1-Y) * tf.log(1 - hypo))

train = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost)

pred = tf.cast(hypo > 0.5, dtype=tf.float32)
accu = tf.reduce_mean(tf.cast(tf.equal(pred, Y), dtype=tf.float32))

with tf.Session() as sess:

    sess.run(tf.global_variables_initializer())

    for step in range(20001):

        sess.run(train, feed_dict={X:x_data, Y :y_data})

        if step % 200 == 0:
            print(step, sess.run(cost, feed_dict={Y:y_data, X:x_data}), sess.run(W))


    # train complete
    h, c, a = sess.run([hypo, pred, accu], feed_dict={X: x_data, Y:y_data})

    print(h, c, a)





