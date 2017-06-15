# 데이터를 파일로 읽어와서
# 학습하는 로직 구현

import tensorflow as tf
import numpy as np
import codecs

# unpack = true 일경우 데이터를 세로로 읽음, float 형태로 읽어들임
# txt파일이 utf8 인코딩이면 안읽어짐 ANSI 로 변경
# s = codecs.open( './train.txt', encoding='utf-8' ).read()
# xy = np.frombuffer(s, dtype="<U2")
# xy = np.loadtxt( './train.txt', unpack=True, dtype='|U10,<U10,float32')
xy = np.loadtxt( './train2.txt', delimiter=',', dtype=np.float32)
x_data = xy[:,0:2]
y_data = xy[:,[-1]]


print(x_data, y_data)

X = tf.placeholder(tf.float32, shape=[None, 2])
Y = tf.placeholder(tf.float32, shape=[None, 1])

W = tf.Variable(tf.random_normal([2,1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')

h = tf.matmul(W, X)
# sigmoid를 가설에 적용함
# hypothesis = tf.div(1., 1.+tf.exp(-h))
hypothesis = tf.sigmoid(tf.matmul(X, W) + b)

# 0 ~ 1 값을 얻기위한 sigmoid가 적용된 코스트함수를 적용한다
cost = -tf.reduce_mean(Y * tf.log(hypothesis) + (1-Y)*tf.log(1-hypothesis))

train = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost)
# 정확도 계산 공식
predicted = tf.cast(hypothesis > 0.5, dtype=tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, Y), dtype=tf.float32))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for step in range(10001):
        cost_val, _ = sess.run( [cost, train], feed_dict={X:x_data, Y:y_data} )
        if step % 100 == 0:
            print( step, cost_val )

    h, c, a = sess.run([hypothesis, predicted, accuracy], feed_dict={X: x_data, Y: y_data})
    print('\nHypothesis:: ', h, "\nCorrect (Y): ", c, '\nAccuracy: ', a)

    # 실제 학습데이터를 이용
    print('----------------------------------------------')
    # 이미 학습된 데이터를 가지고 가설을 돌린다 -> 가설(X)값이 곧 Y 이므로
    # 배열설명 : b, 수업시간, 출석일수
    # 0.5 보다 크면 합격으로 판단
    # 2명을 동시에 적용
    print('apply : ', sess.run(hypothesis, feed_dict={X: [[1, 1], [5, 3], [7, 5]]}) > 0.5)
    # 정확도 계산하기








