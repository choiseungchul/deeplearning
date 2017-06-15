import tensorflow as tf
import numpy as np

tf.set_random_seed(777)

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
X = tf.placeholder(tf.float32, [None, 2])
Y = tf.placeholder(tf.float32, [None, 1])

W1 = tf.Variable(tf.random_normal([2, 2]), name='weight1')
b1 = tf.Variable(tf.random_normal([2]), name='bias1')
layer1 = tf.sigmoid(tf.matmul(X, W1) + b1)

W2 = tf.Variable(tf.random_normal([2, 1]), name='weight2')
b2 = tf.Variable(tf.random_normal([1]), name='bias2')
hypo = tf.sigmoid(tf.matmul(layer1, W2) + b2)

cost = -tf.reduce_mean(Y * tf.log(hypo) + (1-Y) * tf.log(1 - hypo))

train = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)

pred = tf.cast(hypo > 0.5, dtype=tf.float32)
accu = tf.reduce_mean(tf.cast(tf.equal(pred, Y), dtype=tf.float32))

with tf.Session() as sess:

    sess.run(tf.global_variables_initializer())

    for step in range(10001):

        sess.run(train, feed_dict={X:x_data, Y :y_data})

        if step % 200 == 0:
            print(step, sess.run(cost, feed_dict={Y:y_data, X:x_data}), sess.run([W1, W2]))


    # train complete
    h, c, a = sess.run([hypo, pred, accu], feed_dict={X: x_data, Y:y_data})

    print('h:',h, 'cost:',c, "accuracy:",a)



'''
h: [[ 0.01338218]
 [ 0.98809403]
 [ 0.98166394]
 [ 0.01135799]] cost: [[ 0.]
 [ 1.]
 [ 1.]
 [ 0.]] accuracy: 1.0
'''

