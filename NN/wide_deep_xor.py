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

W1 = tf.Variable(tf.random_normal([2, 10]), name='weight1')
b1 = tf.Variable(tf.random_normal([10]), name='bias1')
L1 = tf.sigmoid(tf.matmul(X, W1) + b1)

W2 = tf.Variable(tf.random_normal([10, 10]), name='weight2')
b2 = tf.Variable(tf.random_normal([10]), name='bias2')
L2 = tf.sigmoid(tf.matmul(L1, W2) + b2)

W3 = tf.Variable(tf.random_normal([10, 10]), name='weight3')
b3 = tf.Variable(tf.random_normal([10]), name='bias3')
L3 = tf.sigmoid(tf.matmul(L2, W3) + b3)

W4 = tf.Variable(tf.random_normal([10, 1]), name='weight4')
b4 = tf.Variable(tf.random_normal([1]), name='bias4')
hypo = tf.sigmoid(tf.matmul(L3, W4) + b4)

cost = -tf.reduce_mean(Y * tf.log(hypo) + (1-Y) * tf.log(1 - hypo))

train = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)

pred = tf.cast(hypo > 0.5, dtype=tf.float32)
accu = tf.reduce_mean(tf.cast(tf.equal(pred, Y), dtype=tf.float32))

with tf.Session() as sess:

    sess.run(tf.global_variables_initializer())

    for step in range(4000):

        sess.run(train, feed_dict={X:x_data, Y :y_data})

        if step % 200 == 0:
            print(step, sess.run(cost, feed_dict={Y:y_data, X:x_data}), sess.run([W1, W2]))


    # train complete
    h, c, a = sess.run([hypo, pred, accu], feed_dict={X: x_data, Y:y_data})

    print('h:',h, 'cost:',c, "accuracy:",a)



'''
h: [[ 0.02638706]
 [ 0.95056695]
 [ 0.96523017]
 [ 0.05387093]] cost: [[ 0.]
 [ 1.]
 [ 1.]
 [ 0.]] accuracy: 1.0
'''

