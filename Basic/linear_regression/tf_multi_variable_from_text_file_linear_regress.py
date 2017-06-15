# 데이터를 파일로 읽어와서
# 학습하는 로직 구현

import tensorflow as tf
import numpy as np

# unpack = true 일경우 데이터를 세로로 읽음, float 형태로 읽어들임
# txt파일이 utf8 인코딩이면 안읽어짐 ANSI 로 변경
# s = codecs.open( './train.txt', encoding='utf-8' ).read()
# xy = np.frombuffer(s, dtype="<U2")
# xy = np.loadtxt( './train.txt', unpack=True, dtype='|U10,<U10,float32')
xy = np.loadtxt( 'data-01-test-score.csv', delimiter=',', dtype=np.float32)
x_data = xy[:, 0:-1]
y_data = xy[:,[-1]]

print(x_data.shape, x_data, len(x_data))
print(y_data.shape, y_data)

X = tf.placeholder(tf.float32, shape=[None, 3]);
Y = tf.placeholder(tf.float32, shape=[None, 1]);

W = tf.Variable( tf.random_normal([3,1]), name='weight')
b = tf.Variable( tf.random_normal([1]), name='bias')

hypothesis = tf.matmul(X, W) + b

cost = tf.reduce_mean( tf.square( hypothesis - Y ) )

optimizer = tf.train.GradientDescentOptimizer(1e-5)
train = optimizer.minimize(cost)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for step in range(2001):
    cost_val, hy_val, _ = sess.run(
        [cost, hypothesis, train],
        feed_dict={X:x_data, Y:y_data}
    )
    if step % 10 == 0:
        print( step, "Cost: ", cost_val, "\nPrediction:\n", hy_val )

print( sess.run( hypothesis, feed_dict={ X:[[100,70,100],[60,70,110],[90,100,80]]} ) )


